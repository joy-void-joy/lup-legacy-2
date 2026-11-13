"""A repository's git, run in a room of the repository's own for one command, where no session's room is there to run it (`docs/launch.md`, *A repository's git in a room of its own*).

The host makes and removes a spawned session's worktree with the repository's
own git in a room, never with its own git: a checkout runs the filters a
commit's `.gitattributes` picks, and its writes into `tree/<branch>` follow the
links a room planted on the way, so the host's would land wherever those lead.
In a room, the same links lead only around what the room mounts. A session's
room is often not there to run it: another repository has none of the
spawner's, a restart may come after its spawner ended, and a side job's
worktree goes once its spawner's room has. So the host starts a room for the
one command (`Podman.once`):
- **its image** is the repository's room image built last, found by its
  project's label, so the git it runs is the one the repository's rooms run,
  and nothing is built for it: where none is built, nothing runs, and the
  caller is told a launch there builds one;
- **it mounts** the repository's git directory alone, read-write, where every
  worktree lup makes is (`made_root`), with its `config` and `hooks/` read-only
  over it, as every room of the repository mounts them: nothing of the
  operator's home, of the repository's other worktrees, or of another
  repository;
- **its git** reads no configuration but the repository's own `config`, which
  no room writes (`GIT_CONFIG_GLOBAL=/dev/null`, `GIT_CONFIG_NOSYSTEM`), its
  home the image's root, which it can't write, so no ignore or attribute file
  a room planted is read as its own; on no network, so nothing it does
  fetches, pushes or reaches the host's credentials;
- **its time** is held by podman's own monitor past `within` (`--timeout`), and
  the host's wait on it past `waiting` more, after which the host removes the
  container by its name: a room can make a ref a pipe nothing writes, in which
  its git would wait for good.
"""

import secrets
from collections.abc import Awaitable, Callable
from datetime import timedelta

from pydantic import TypeAdapter, ValidationError

from lup.programs import Ran
from lup.rooms.container import AbsolutePath, Mount
from lup.rooms.engine import PROJECT_LABEL, Once, Podman, StoredImage
from lup.types import Model
from lup_dev.errors import LupDevError
from lup_dev.host.service import Program, executed


class GitRoomError(LupDevError):
    """A repository's git that couldn't be run in a room of its own, and why."""


class GitRoom(Model, arbitrary_types_allowed=True):
    """A repository's git, run in a room of its own for one command at a time."""

    repository: AbsolutePath
    """Its common git directory, as the host recorded it."""
    run: Callable[[Program], Awaitable[Ran]] = executed
    """Runs one program on the host, killed past its `within` with what it
    started, ending with status 124."""
    podman: Podman = Podman()
    within: timedelta = timedelta(minutes=2)
    """How long its git may take, a checkout of the whole tree among it, before
    podman's monitor kills its room."""
    waiting: timedelta = timedelta(seconds=30)
    """How much longer than `within` the host waits on the podman that started
    it, before killing that podman and removing the room by its name."""
    asking: timedelta = timedelta(seconds=30)
    """How long podman may take to list the repository's images, or remove its
    room."""
    heard: int = 16 * 1024
    """The most of what its git prints that's kept, its end: a room chooses
    what that git reads, which could print without end into the service's
    memory (`Program.kept`)."""

    async def image(self) -> str:
        """Return the id of the repository's room image built last; raise `GitRoomError` where none is, or podman can't say."""
        listed = await self.run(
            Program(
                command=[
                    self.podman.program,
                    "images",
                    "--filter",
                    f"label={PROJECT_LABEL}={self.repository}",
                    "--format",
                    "json",
                ],
                payload="",
                within=self.asking,
            )
        )
        if listed.status != 0:
            said = listed.said() or f"it ended with status {listed.status}"
            message = (
                f"podman couldn't list the room images of {self.repository}: {said}"
            )
            raise GitRoomError(message)
        try:
            images = TypeAdapter(list[StoredImage]).validate_json(listed.output)
        except ValidationError as unread:
            message = f"podman's list of the room images of {self.repository} didn't read: {unread}"
            raise GitRoomError(message) from unread
        if not images:
            message = (
                f"no room image of {self.repository} is built, which its git runs "
                "in: a launch there builds one"
            )
            raise GitRoomError(message)
        return max(images, key=lambda each: each.created).id

    async def git(self, arguments: list[str]) -> Ran:
        """Run git with `arguments` on the repository, in a room of its own (`Podman.once`), and return how it ended, whatever its status: 124 where it ran past `within`. Raise `GitRoomError` where no room can be started for it."""
        image = await self.image()
        hooks = self.repository / "hooks"
        hooks.mkdir(exist_ok=True)
        name = f"lup-git-{secrets.token_hex(8)}"
        once = Once(
            image=image,
            mounts=[
                Mount(source=self.repository),
                Mount(source=self.repository / "config", read_only=True),
                Mount(source=hooks, read_only=True),
            ],
            # Its home is the image's root, which its user can't write: podman
            # would otherwise make it the working directory, the git directory
            # a room writes, where git reads ignore and attribute files.
            env={
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_NOSYSTEM": "1",
                "HOME": "/",
            },
            workdir=self.repository,
            command=["git", "--git-dir", str(self.repository), *arguments],
            name=name,
            within=self.within,
        )
        ran = await self.run(
            Program(
                command=self.podman.once(once),
                payload="",
                directory=self.repository,
                within=self.within + self.waiting,
                kept=self.heard,
            )
        )
        if ran.status == 124:  # `executed` killed the podman that started it
            await self.removed(name)
        return ran

    async def removed(self, name: str) -> None:
        """Remove the room named `name`, whatever runs in it, where the podman that started it was killed: its monitor kills it in time all the same, and this makes sure."""
        await self.run(
            Program(
                command=[self.podman.program, "rm", "--force", "--ignore", name],
                payload="",
                within=self.asking,
            )
        )
