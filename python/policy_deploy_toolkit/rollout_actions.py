"""Simulation-time action sequences for scripted policy rollouts.

Siblings run sequentially; start_offset is an extra delay after the preceding
sibling. Children start relative to their parent and run while it remains
active. Pass ActionSequence(actions) to Simulation.run(before_step=...).
"""

from dataclasses import dataclass, field
import math
from typing import Callable

__all__ = ['Action', 'EmptyAction', 'VelocityCommand', 'ApplyForce', 'Respawn', 'ActionSequence']

_EPS = 1e-9


def _finite(value, name):
    if isinstance(value, bool):
        raise ValueError(f'{name} must be finite')
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be finite') from exc
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def _vector3(value, name):
    try:
        result = tuple(_finite(item, name) for item in value)
    except TypeError as exc:
        raise ValueError(f'{name} must contain three finite numbers') from exc
    if len(result) != 3:
        raise ValueError(f'{name} must contain three finite numbers')
    return result


def _span(actions):
    try:
        total = math.fsum(part for action in actions for part in (action.start_offset, action.duration))
    except OverflowError as exc:
        raise ValueError('Action durations and offsets must have a finite sum') from exc
    if not math.isfinite(total):
        raise ValueError('Action durations and offsets must have a finite sum')
    return total


@dataclass(frozen=True, kw_only=True)
class Action:
    """An immutable action definition, reusable in multiple sequences.

    Override start() to perform an effect and optionally return a cleanup
    callback. The scheduler keeps runtime state separately from definitions.
    """

    duration: float = 1.
    start_offset: float = 0.
    sub_actions: tuple['Action', ...] = field(default_factory=tuple)

    def __post_init__(self):
        duration = _finite(self.duration, 'duration')
        offset = _finite(self.start_offset, 'start_offset')
        if duration < 0 or offset < 0:
            raise ValueError('duration and start_offset must be non-negative')
        children = tuple(self.sub_actions)
        if any(not isinstance(child, Action) for child in children):
            raise TypeError('sub_actions must contain Action instances')
        span = _span(children)
        if span > duration and (duration == 0 or not math.isclose(span, duration, abs_tol=_EPS, rel_tol=0)):
            raise ValueError('sub_actions must fit within the parent duration')
        object.__setattr__(self, 'duration', duration)
        object.__setattr__(self, 'start_offset', offset)
        object.__setattr__(self, 'sub_actions', children)

    def start(self, sim, remaining_duration: float) -> Callable[[], None] | None:
        """Start once; remaining_duration excludes any delay to the callback tick."""
        return None

    @property
    def description(self):
        return type(self).__name__


@dataclass(frozen=True, kw_only=True)
class EmptyAction(Action):
    """Wait without changing commands; a parent action continues during the wait."""

    @property
    def description(self):
        return 'Wait'


@dataclass(frozen=True, kw_only=True)
class VelocityCommand(Action):
    """Override a velocity command for duration, then restore its previous value."""

    vx: float = 0.
    vy: float = 0.
    wz: float = 0.
    command_name: str = 'base_velocity'

    def __post_init__(self):
        super().__post_init__()
        if self.duration <= 0:
            raise ValueError('VelocityCommand duration must be positive')
        for name in ('vx', 'vy', 'wz'):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not isinstance(self.command_name, str) or not self.command_name:
            raise ValueError('command_name must be a nonempty string')

    def start(self, sim, remaining_duration):
        if self.command_name not in sim.commands:
            raise ValueError(f'Policy has no command {self.command_name!r}')
        previous = sim.commands[self.command_name].copy()
        sim.set_command(self.command_name, [self.vx, self.vy, self.wz])
        return lambda: sim.set_command(self.command_name, previous)

    @property
    def description(self):
        return f'Velocity vx={self.vx:g} vy={self.vy:g} m/s, wz={self.wz:g} rad/s'


@dataclass(frozen=True, kw_only=True)
class ApplyForce(Action):
    """Add a timed COM force (N), optionally with torque (Nm), on a named body."""

    fx: float = 0.
    fy: float = 0.
    fz: float = 0.
    body: str = 'base_link'
    frame: str = 'body'
    torque: tuple[float, float, float] = (0., 0., 0.)

    def __post_init__(self):
        super().__post_init__()
        if self.duration <= 0:
            raise ValueError('ApplyForce duration must be positive')
        for name in ('fx', 'fy', 'fz'):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not isinstance(self.body, str) or not self.body:
            raise ValueError('body must be a nonempty body name')
        if self.frame not in ('global', 'local', 'world', 'body'):
            raise ValueError('frame must be global/world or local/body')
        object.__setattr__(self, 'torque', _vector3(self.torque, 'torque'))

    def start(self, sim, remaining_duration):
        # Timed wrenches add and expire in the main simulator's control manager.
        # Clearing by body here would also remove other overlapping wrenches.
        sim.apply_wrench(self.body, force=[self.fx, self.fy, self.fz], torque=self.torque,
                         frame=self.frame, duration=remaining_duration)

    @property
    def description(self):
        text = f'Force {self.body}: ({self.fx:g}, {self.fy:g}, {self.fz:g}) N [{self.frame}]'
        if any(self.torque):
            text += ' T=(' + ', '.join(f'{value:g}' for value in self.torque) + ') Nm'
        return text


@dataclass(frozen=True, kw_only=True)
class Respawn(Action):
    """Instantaneous respawn at a world position and yaw in radians."""

    position: tuple[float, float, float]
    yaw: float = 0.
    duration: float = field(default=0., init=False)

    def __post_init__(self):
        super().__post_init__()
        if self.sub_actions:
            raise ValueError('Respawn cannot contain sub_actions')
        object.__setattr__(self, 'position', _vector3(self.position, 'position'))
        object.__setattr__(self, 'yaw', _finite(self.yaw, 'yaw'))

    def start(self, sim, remaining_duration):
        sim.respawn(position=self.position,
                    quaternion=[math.cos(self.yaw / 2), 0., 0., math.sin(self.yaw / 2)])

    @property
    def description(self):
        return f'Respawn {self.position}, yaw={self.yaw:g} rad'


class ActionSequence:
    """Callable scheduler for one rollout, driven by sim.elapsed_time.

    Effects start on the first policy callback at or after their scheduled time.
    Already-expired positive-duration actions are skipped rather than replayed.
    Instantaneous actions still execute once when their time is crossed.
    Returns False after all actions (including final instantaneous ones) finish.
    Construct a new sequence to reuse definitions in another rollout.
    """

    def __init__(self, actions):
        self.actions = tuple(actions)
        if any(not isinstance(action, Action) for action in self.actions):
            raise TypeError('actions must contain Action instances')
        self.duration = _span(self.actions)
        self._events = []
        self._cleanups = {}
        self._active = {}
        self._origin = None
        self._sim = None
        self._last_time = 0.
        self._index = 0

        def schedule(action, start):
            token = len(self._events)
            end = start + action.duration
            self._events.append((start, len(self._events), True, token, action, end))
            cursor = start
            for child in action.sub_actions:
                cursor = math.fsum((cursor, child.start_offset))
                schedule(child, cursor)
                cursor = math.fsum((cursor, child.duration))
            self._events.append((end, len(self._events), False, token, action, end))

        cursor = 0.
        for action in self.actions:
            cursor = math.fsum((cursor, action.start_offset))
            schedule(action, cursor)
            cursor = math.fsum((cursor, action.duration))
        self._events.sort(key=lambda event: event[0])
        # Preserve tree traversal order at boundaries despite floating-point
        # roundoff: parent starts first; children finish before parent cleanup.
        normalized = []
        index = 0
        while index < len(self._events):
            end = index + 1
            boundary = self._events[index][0]
            while end < len(self._events) and self._events[end][0] - boundary <= _EPS:
                end += 1
            normalized.extend(sorted(self._events[index:end], key=lambda event: event[1]))
            index = end
        self._events = normalized

    @property
    def finished(self):
        return self._index == len(self._events)

    @property
    def status_text(self):
        if self.finished:
            return 'Actions complete'
        if not self._active:
            return 'Waiting for action'
        return '\n'.join(f'{action.description} ({max(0., end - self._last_time):.1f}s)'
                         for action, end in self._active.values())

    def __call__(self, sim):
        current = _finite(sim.elapsed_time, 'sim.elapsed_time')
        if self._origin is None:
            self._origin = current
            self._sim = sim
        elif sim is not self._sim:
            raise ValueError('An ActionSequence can only drive one simulation')
        now = current - self._origin
        if now < self._last_time - _EPS:
            raise ValueError('sim.elapsed_time must not decrease; use the session clock')
        self._last_time = now
        while self._index < len(self._events):
            timestamp, _, starting, token, action, end = self._events[self._index]
            if timestamp > now + _EPS:
                break
            if starting:
                if action.duration == 0 or now < end - _EPS:
                    cleanup = action.start(sim, max(0., end - now))
                    self._active[token] = (action, end)
                    if cleanup is not None:
                        self._cleanups[token] = cleanup
            else:
                self._active.pop(token, None)
                cleanup = self._cleanups.pop(token, None)
                if cleanup is not None:
                    cleanup()
            self._index += 1
        if callable(getattr(sim, 'set_status_text', None)):
            sim.set_status_text(self.status_text)
        return not self.finished
