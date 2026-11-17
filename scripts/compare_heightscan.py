#!/usr/bin/env python3
"""Height scan errors against the MuJoCo ray-cast ground truth, as heights below the base in the
gravity-aligned frame (what the policy sees after leveling):
  total: the lio_heightscan.py output
  map:   the elevation map sampled at the true pose (odom <- world fixed at the first sample)
  pose:  total - map
Points are split into edge (a grid neighbour differs by more than edge_threshold) and flat.
Run from this directory: python3 compare_heightscan.py --ros-args -p world_xml:=... -p metadata:=...
"""
import sys

import numpy as np
import rclpy
from grid_map_msgs.msg import GridMap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

import sim_heightscan as gt
from lio_heightscan import HeightLayer, quat_to_mat, scan_points, yaw_of


def se3(p, q):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(q.x, q.y, q.z, q.w)
    T[:3, 3] = (p.x, p.y, p.z)
    return T


def edge_mask(z, shape, threshold):
    """Cells with a 4-neighbour more than threshold apart, on the scan grid."""
    g = z.reshape(shape)
    m = np.zeros(shape, bool)
    dx = np.abs(np.diff(g, axis=1)) > threshold
    dy = np.abs(np.diff(g, axis=0)) > threshold
    m[:, 1:] |= dx
    m[:, :-1] |= dx
    m[1:, :] |= dy
    m[:-1, :] |= dy
    return m.ravel()


def stats(e):
    if len(e) == 0:
        return 'no points'
    a = np.abs(e) * 100
    return f'mean {a.mean():4.1f}  p95 {np.percentile(a, 95):4.1f}  max {a.max():4.1f}  bias {e.mean() * 100:+5.1f} cm'


def main():
    rclpy.init(args=sys.argv)
    node = rclpy.create_node('compare_heightscan')
    world_xml = node.declare_parameter('world_xml', '').value
    metadata = node.declare_parameter('metadata', '').value
    pose_topic = node.declare_parameter('pose_topic', '/pelvis/odom').value
    odom_topic = node.declare_parameter('odom_topic', '/localization/fast_lio_odom_bridge/odometry').value
    map_topic = node.declare_parameter('map_topic', '/perception/elevation_mapping_node/elevation_map').value
    scan_topic = node.declare_parameter('scan_topic', '/heightscan').value
    layer_name = node.declare_parameter('layer', 'inpaint').value
    edge_threshold = node.declare_parameter('edge_threshold', 0.03).value
    period = node.declare_parameter('period', 2.0).value
    output = node.declare_parameter('output', '/home/user/forest_ws/heightscan_compare.npz').value
    log = node.get_logger()

    scanner = gt.load_scanner(metadata)
    cfg = scanner['pattern_cfg']
    grid = gt.grid_pattern(cfg)
    offset = np.array(scanner['offset']['pos'], dtype=float)
    nx, ny = len(np.unique(grid[:, 0].round(6))), len(np.unique(grid[:, 1].round(6)))
    shape = (ny, nx) if cfg['ordering'] == 'xy' else (nx, ny)

    model = gt.mujoco.MjModel.from_xml_path(world_xml)
    data = gt.mujoco.MjData(model)
    gt.mujoco.mj_forward(model, data)
    model.geom_group[model.body_weldid[model.geom_bodyid] != 0] = 5
    groups = np.array([1, 1, 1, 1, 1, 0], np.uint8)

    latest = {}
    acc = {k: [] for k in ('total', 'map', 'pose', 'edge', 'pose_xy', 'pose_z', 'pose_yaw')}
    rec = {k: [] for k in ('t', 'z_gt', 'z_lio', 'z_map', 'pos', 'pose_xy', 'pose_z', 'pose_yaw')}

    def on_scan(msg):
        missing = [t for k, t in (('pose', pose_topic), ('odom', odom_topic), ('map', map_topic)) if k not in latest]
        if missing:
            log.info(f'Waiting for {", ".join(missing)}', throttle_duration_sec=5.0)
            return
        T_wb = se3(latest['pose'].pose.pose.position, latest['pose'].pose.pose.orientation)
        T_ob_lio = se3(latest['odom'].pose.pose.position, latest['odom'].pose.pose.orientation)
        if 'T_ow' not in latest:
            latest['T_ow'] = T_ob_lio @ np.linalg.inv(T_wb)
            log.info('odom <- world fixed from the current LIO and true poses')
        pos, rot = T_wb[:3, 3], T_wb[:3, :3]

        z_gt = (gt.scan_points(model, data, groups, grid, offset, pos, rot) @ rot.T)[:, 2]
        p_lio = point_cloud2.read_points_numpy(msg, field_names=['x', 'y', 'z']).astype(float)
        if len(p_lio) != len(grid):
            log.error(f'{scan_topic} has {len(p_lio)} points, grid {len(grid)}', throttle_duration_sec=5.0)
            return
        z_total = (p_lio @ rot.T)[:, 2]
        T_ob = latest['T_ow'] @ T_wb
        pts, _ = scan_points(latest['map'], grid, offset, T_ob[:3, 3], T_ob[:3, :3], 0.562)
        z_map = (pts @ T_ob[:3, :3].T)[:, 2]

        acc['total'].append(z_total - z_gt)
        acc['map'].append(z_map - z_gt)
        acc['pose'].append(z_total - z_map)
        acc['edge'].append(edge_mask(z_gt, shape, edge_threshold))
        T_wb_lio = np.linalg.inv(latest['T_ow']) @ T_ob_lio
        acc['pose_xy'].append(np.linalg.norm(T_wb_lio[:2, 3] - pos[:2]))
        acc['pose_z'].append(T_wb_lio[2, 3] - pos[2])
        acc['pose_yaw'].append(np.degrees(yaw_of(T_wb_lio[:3, :3]) - yaw_of(rot) + np.pi) % 360 - 180)
        stamp = msg.header.stamp
        for k, v in (('t', stamp.sec + stamp.nanosec * 1e-9), ('z_gt', z_gt), ('z_lio', z_total), ('z_map', z_map),
                     ('pos', pos.copy()), ('pose_xy', acc['pose_xy'][-1]), ('pose_z', acc['pose_z'][-1]),
                     ('pose_yaw', acc['pose_yaw'][-1])):
            rec[k].append(v)

    run = {k: [] for k in acc}

    def report(window, title=''):
        if not window['total']:
            return
        edge = np.concatenate(window['edge'])
        lines = [f'{title}{len(window["total"])} scans, {edge.mean():.0%} edge points']
        for k in ('total', 'map', 'pose'):
            e = np.concatenate(window[k])
            lines.append(f'  {k:5s} flat: {stats(e[~edge])}   edge: {stats(e[edge])}')
        lines.append(f'  LIO pose vs true: xy mean {np.mean(window["pose_xy"]) * 100:.1f} cm, '
                     f'z mean {np.mean(window["pose_z"]) * 100:+.1f} cm, yaw mean {np.mean(window["pose_yaw"]):+.2f} deg')
        print('\n'.join(lines), flush=True)

    def report_period():
        report(acc)
        for k, v in acc.items():
            run[k].extend(v)
            v.clear()

    node.create_subscription(Odometry, pose_topic, lambda m: latest.update(pose=m), 1)
    node.create_subscription(Odometry, odom_topic, lambda m: latest.update(odom=m), 10)
    node.create_subscription(GridMap, map_topic, lambda m: latest.update(map=HeightLayer(m, layer_name)), 1)
    node.create_subscription(PointCloud2, scan_topic, on_scan, 10)
    node.create_timer(period, report_period)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    for k, v in acc.items():
        run[k].extend(v)
    report(run, 'WHOLE RUN: ')
    if rec['t']:
        np.savez(output, shape=shape, edge_threshold=edge_threshold, **{k: np.array(v) for k, v in rec.items()})
        print(f'Saved {len(rec["t"])} scans to {output}; plot: python3 compare_heightscan.py --plot {output}')


def plot(path):
    """Error over time, LIO pose error and the true path, and the worst scan as GT / LIO / error grids."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = np.load(path)
    t = d['t'] - d['t'][0]
    shape = tuple(d['shape'])
    edge = np.array([edge_mask(z, shape, float(d['edge_threshold'])) for z in d['z_gt']])
    err = {'total': d['z_lio'] - d['z_gt'], 'map': d['z_map'] - d['z_gt'], 'pose': d['z_lio'] - d['z_map']}

    fig, ax = plt.subplots(3, 3, figsize=(18, 12))
    for k, c in zip(err, ('k', 'tab:blue', 'tab:orange')):
        a = np.abs(err[k]) * 100
        ax[0, 0].plot(t, np.nanmean(np.where(edge, np.nan, a), axis=1), c, label=k)
        ax[0, 1].plot(t, np.nanmean(np.where(edge, a, np.nan), axis=1), c, label=k)
        ax[0, 2].plot(t, a.max(axis=1), c, label=k)
    for a, title in zip(ax[0], ('mean |error| flat points [cm]', 'mean |error| edge points [cm]', 'max |error| [cm]')):
        a.set_title(title)
        a.set_xlabel('t [s]')
        a.legend()
        a.grid()
    ax[1, 0].plot(t, d['pose_xy'] * 100, label='xy [cm]')
    ax[1, 0].plot(t, d['pose_z'] * 100, label='z [cm]')
    ax[1, 0].plot(t, d['pose_yaw'], label='yaw [deg]')
    ax[1, 0].set_title('LIO pose - true')
    ax[1, 0].set_xlabel('t [s]')
    ax[1, 0].legend()
    ax[1, 0].grid()
    worst = int(np.argmax(np.abs(err['total']).max(axis=1)))
    sc = ax[1, 1].scatter(d['pos'][:, 0], d['pos'][:, 1], c=np.abs(err['total']).mean(axis=1) * 100, s=4)
    ax[1, 1].plot(*d['pos'][worst, :2], 'rx', ms=12)
    ax[1, 1].set_title('true path, colour: mean |total error| [cm], x: worst scan')
    ax[1, 1].axis('equal')
    fig.colorbar(sc, ax=ax[1, 1])
    ax[1, 2].hist([np.abs(e[~edge]) * 100 for e in err.values()], bins=50, label=list(err), histtype='step')
    ax[1, 2].set_title('|error| flat points [cm]')
    ax[1, 2].set_yscale('log')
    ax[1, 2].legend()
    g, l = d['z_gt'][worst].reshape(shape), d['z_lio'][worst].reshape(shape)
    lo, hi = min(g.min(), l.min()), max(g.max(), l.max())
    for a, img, title in ((ax[2, 0], g, 'GT height below base [m]'), (ax[2, 1], l, 'LIO height below base [m]')):
        fig.colorbar(a.imshow(img, origin='lower', vmin=lo, vmax=hi), ax=a)
        a.set_title(f'{title}, worst scan t={t[worst]:.1f} s')
    e = (l - g) * 100
    m = np.abs(e).max()
    fig.colorbar(ax[2, 2].imshow(e, origin='lower', cmap='RdBu_r', vmin=-m, vmax=m), ax=ax[2, 2])
    ax[2, 2].set_title('LIO - GT [cm] (rows: lateral, columns: forward)')
    fig.tight_layout()
    out = path.rsplit('.', 1)[0] + '.png'
    fig.savefig(out, dpi=80)
    print(f'Saved {out}')


def video(path, fps=10):
    """GIF of GT and LIO patches, their difference and the centre-line profile, one frame per 1/fps sim s."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    d = np.load(path)
    t = d['t'] - d['t'][0]
    shape = tuple(d['shape'])
    frames = np.unique(np.searchsorted(t, np.arange(0.0, t[-1], 1.0 / fps)))
    gt_all, lio_all = d['z_gt'].reshape(-1, *shape), d['z_lio'].reshape(-1, *shape)
    lo, hi = min(gt_all.min(), lio_all.min()), max(gt_all.max(), lio_all.max())
    centre = shape[0] // 2
    forward = np.arange(shape[1])

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    im_gt = ax[0, 0].imshow(gt_all[0], origin='lower', vmin=lo, vmax=hi)
    im_lio = ax[0, 1].imshow(lio_all[0], origin='lower', vmin=lo, vmax=hi)
    im_err = ax[1, 0].imshow((lio_all[0] - gt_all[0]) * 100, origin='lower', cmap='RdBu_r', vmin=-10, vmax=10)
    for a, title in ((ax[0, 0], 'GT [m below base]'), (ax[0, 1], 'LIO [m below base]'),
                     (ax[1, 0], 'LIO - GT [cm]')):
        a.set_title(title)
        a.axhline(centre, color='w', lw=0.5)
        a.set_xlabel('forward cell')
        a.set_ylabel('lateral cell')
    fig.colorbar(im_gt, ax=ax[0, :])
    fig.colorbar(im_err, ax=ax[1, 0])
    (l_gt,) = ax[1, 1].plot(forward, gt_all[0, centre], 'k.-', label='GT')
    (l_lio,) = ax[1, 1].plot(forward, lio_all[0, centre], 'r.-', label='LIO')
    ax[1, 1].set_ylim(lo - 0.02, hi + 0.02)
    ax[1, 1].set_title('centre line profile')
    ax[1, 1].set_xlabel('forward cell')
    ax[1, 1].legend()
    ax[1, 1].grid()

    def draw(i):
        im_gt.set_data(gt_all[i])
        im_lio.set_data(lio_all[i])
        im_err.set_data((lio_all[i] - gt_all[i]) * 100)
        l_gt.set_ydata(gt_all[i, centre])
        l_lio.set_ydata(lio_all[i, centre])
        fig.suptitle(f't = {t[i]:.1f} s   mean |LIO - GT| {np.abs(lio_all[i] - gt_all[i]).mean() * 100:.1f} cm')
        return im_gt, im_lio, im_err, l_gt, l_lio

    out = path.rsplit('.', 1)[0] + '.gif'
    FuncAnimation(fig, draw, frames=frames).save(out, writer=PillowWriter(fps=fps), dpi=70)
    print(f'Saved {out} ({len(frames)} frames)')


def compare(paths, bin_s=1.0):
    """Total error of several runs on the same bag: table sorted by edge p95, bars, error over sim time,
    and the centre-line profiles at the worst frame of the best run."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    runs = []
    for p in paths:
        d = np.load(p)
        shape = tuple(d['shape'])
        edge = np.array([edge_mask(z, shape, float(d['edge_threshold'])) for z in d['z_gt']])
        err = d['z_lio'] - d['z_gt']
        name = os.path.splitext(os.path.basename(p))[0].removeprefix('cmp_')
        a = np.abs(err)
        runs.append(dict(name=name, d=d, shape=shape, edge=edge, err=err,
                         flat=err[~edge], edges=err[edge], edge_p95=np.percentile(a[edge], 95) if edge.any() else np.nan))
    runs.sort(key=lambda r: r['edge_p95'])

    print(f'{"run":24s} {"scans":>6s} | {"flat: mean   p95   max  bias":>30s} | {"edge: mean   p95   max  bias":>30s}  [cm]')
    for r in runs:
        cols = []
        for e in (r['flat'], r['edges']):
            a = np.abs(e) * 100
            cols.append(f'{a.mean():6.1f} {np.percentile(a, 95):5.1f} {a.max():5.1f} {e.mean() * 100:+5.1f}')
        print(f'{r["name"]:24s} {len(r["err"]):6d} | {cols[0]:>30s} | {cols[1]:>30s}')

    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    x = np.arange(len(runs))
    for k, (key, label) in enumerate((('flat', 'flat'), ('edges', 'edge'))):
        a = [np.abs(r[key]) * 100 for r in runs]
        ax[0].bar(x + (k - 0.5) * 0.4, [v.mean() for v in a], 0.4, label=f'{label} mean')
        ax[0].scatter(x + (k - 0.5) * 0.4, [np.percentile(v, 95) for v in a], marker='_', s=400, c='k',
                      label='p95' if k == 0 else None)
    ax[0].set_xticks(x, [r['name'] for r in runs], rotation=30, ha='right')
    ax[0].set_ylabel('|LIO - GT| [cm]')
    ax[0].set_title('total error (bars: mean, ticks: p95)')
    ax[0].legend()
    ax[0].grid(axis='y')

    t0 = min(r['d']['t'][0] for r in runs)
    for r in runs:
        t = r['d']['t'] - t0
        b = np.floor(t / bin_s).astype(int)
        per_scan = np.abs(r['err']).mean(axis=1) * 100
        sums, counts = np.bincount(b, per_scan), np.bincount(b)
        keep = counts > 0
        ax[1].plot(np.arange(len(sums))[keep] * bin_s, sums[keep] / counts[keep], label=r['name'])
    ax[1].set_xlabel('bag time [s]')
    ax[1].set_ylabel(f'mean |LIO - GT| per scan, {bin_s:g} s bins [cm]')
    ax[1].set_title('error along the path')
    ax[1].legend()
    ax[1].grid()

    best = runs[0]
    worst = int(np.argmax(np.abs(best['err']).max(axis=1)))
    t_worst = best['d']['t'][worst]
    centre = best['shape'][0] // 2
    forward = np.arange(best['shape'][1])
    ax[2].plot(forward, best['d']['z_gt'][worst].reshape(best['shape'])[centre], 'k.-', lw=2, label='GT')
    for r in runs:
        i = int(np.argmin(np.abs(r['d']['t'] - t_worst)))
        ax[2].plot(forward, r['d']['z_lio'][i].reshape(r['shape'])[centre], '.-', label=r['name'])
    ax[2].set_xlabel('forward cell')
    ax[2].set_ylabel('height below base [m]')
    ax[2].set_title(f'centre line at the worst frame of {best["name"]} (t={t_worst - t0:.1f} s)')
    ax[2].legend()
    ax[2].grid()
    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(paths[0])), 'cmp_runs.png')
    fig.savefig(out, dpi=80)
    print(f'Saved {out}')


def selftest():
    z = np.zeros((3, 4))
    z[:, 2:] = 0.2  # step between columns 1 and 2
    m = edge_mask(z.ravel(), z.shape, 0.03).reshape(z.shape)
    assert m[:, 1:3].all() and not m[:, [0, 3]].any()
    print('selftest ok')


if __name__ == '__main__':
    if '--plot' in sys.argv:
        plot(sys.argv[sys.argv.index('--plot') + 1])
    elif '--video' in sys.argv:
        video(sys.argv[sys.argv.index('--video') + 1])
    elif '--compare' in sys.argv:
        compare(sys.argv[sys.argv.index('--compare') + 1:])
    elif '--selftest' in sys.argv:
        selftest()
    else:
        main()
