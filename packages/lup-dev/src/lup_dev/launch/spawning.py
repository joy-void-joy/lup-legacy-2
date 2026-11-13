"""An agent's `spawn`: a session of its own, held with no terminal, in a room and a worktree of its own (`docs/launch.md`, *An agent's spawn*).

The tool is the host service's, served each launched session on its
`mcp.sock` beside the inbox's. One call starts what `lup-dev spawn` starts for
the operator, through the same launcher (`Spawning.started`), and returns at
once; the launch's setup runs behind it. Two kinds, told apart by what's
given rather than by a kind: with a `concern`, it's a concern handed over to
build, which the session spawned holds, and where what it says lands, the
operator talking with it directly; without, a side job for the spawner, whose
report, its turn's end, comes back to them whole (`Delivery.reported`).

**Why a room can't make more of it than that.** A room asks the host to start
something outside it, so what the call can name, and what reaches a command
line, a path or git, is held to this:
- **Who asks** is the conversation lup's hook stamped on the call, of the
  launch whose `mcp.sock` it came on (`LaunchTools`): the socket says the launch,
  never the room. The spawner recorded with the launch (`Spawner`) is that
  launch and that conversation, so the parent link is the host's, not the
  room's word.
- **The repository** is the one the host recorded for the spawner's launch
  (`Register.repository`), never a path the call names. One the call names by
  name is read from what the spawner's repository's approval lets its
  sessions spawn in, at the path the host kept as the operator approved it
  (`spawnable_in`, `Approval.spawnable`), and refused yet: no room that's the
  spawner's could make its worktree. The launcher takes the repository as
  given (`HeldLaunch.repository`, `located` in `cli.py`), never finding it
  through the worktree, whose `.git` file a room writes, and refuses a
  worktree that repository doesn't name back.
- **The worktree** is made at `tree/<branch>` in that repository's git
  directory (`made_root`), never by the host's git on what a room committed: a
  checkout runs filters a commit's `.gitattributes` picks, and writes through
  links a room planted on the way. It's made by the spawner's room's own git,
  through `podman exec` (as `land` removes worktrees), pinned to the git
  directory, its hooks and file monitor off (`git_in`): what it does there,
  that room could do itself. Never in another session's room, where what a
  room planted in the shared git directory would run as that session, nor in
  a room started for the one command, which wouldn't hide what a session's
  room hides. The command is a list, no shell: `git`, the git directory from
  the host's record, `worktree add`, the branch, the root and a commit's id.
- **The branch** is the call's `name` in a branch's shape (`branch_named`):
  ASCII letters, digits and `-`, never starting with `-`, so neither git nor a
  path reads it as anything else; a branch or a worktree that's there already
  takes the next number.
- **What it starts from** is a local branch by its short name, held to a
  branch's shape (`branch_name`, then git's own `check-ref-format` on
  `refs/heads/<it>`, which expands no `@{-1}` as `--branch` would), read by
  git pinned to the common directory (`Git.pinned`, hooks off, no replace
  objects, no commit graph) as `refs/heads/<it>`, peeled to a commit; only
  that commit's id reaches the room's git.
- **The runtime** is the spawner's, and only one with a session held with no
  terminal; the call names none, nor any of its arguments, a profile or an
  account: the launch runs on its repository's default profile, as
  `lup-dev spawn` with none named does.
- **The brief** is text in the inbox alone: an item from the spawner to the
  session spawned, which reaches it as its first turn through its launcher, as
  the operator's words do (`Delivery.called_up`). `concern` is resolved through
  the concerns of the repository it starts in alone: in another, one nobody
  running there answers for, or a new title at its top, which nobody holds
  until it's handed (`handing`).
- **How many:** at most `limit` sessions an agent spawned run at once on the
  host, and `each` of one repository's, counted from those served, those whose
  launcher started and hasn't registered yet, and the spawns under way: against
  a loop of spawns, not a budget.
- **Nothing a room can slow holds the service, or another room's spawn.** git
  on the host reads refs a room writes, any of which it can make a pipe
  nothing writes: it runs in the repository's own threads, off the service's
  loop (`threaded`), killed past `reading`, and a branch's free number is
  read in one call, `numbers` at most tried. `podman exec` running the room's
  own git to make the worktree is killed past `making`, with what it started
  (`executed`), and no more than `killing` of a launch's such commands run or
  were killed at once, each of which can leave its git and a host process of
  podman's running. Nothing is held across either: a spawn is counted, and
  takes its branch, at once, between two awaits (`reserved`).
"""

import asyncio
import secrets
from collections.abc import Awaitable, Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, override

from pydantic import Field, JsonValue

import lup
from lup.programs import ProgramError, Ran
from lup.tools.serving import Answer
from lup.tools.tool import Accepted, AgentTool, ToolSpec
from lup.tools.tool import Refused as Rejected
from lup.types import Model, MutableModel
from lup_dev.approval import approved, approved_branches
from lup_dev.git import Git, GitError, pinning, threaded
from lup_dev.host.held import HeldLaunch, Launchers, Spawning, Starting
from lup_dev.host.protocol import Register, Spawner, branch_name, segment
from lup_dev.host.rooms import RoomTools
from lup_dev.host.service import Program, executed
from lup_dev.inbox.delivery import Delivery
from lup_dev.inbox.store import InboxError, Message
from lup_dev.launch.secrets import git_directory
from lup_dev.layout import Layout
from lup_dev.policy.worktrees import branch_named, made_root
from lup_dev.stored import write_model
from lup_dev.trial import running_tip
from lup_ledger.parties import Conversation, Operator
from lup_ledger.threads import Thread

if TYPE_CHECKING:
    from lup_dev.host.courier import Conversation as Caller


class Spawn(Model):
    """A session to start: its name, its brief, and the concern handed with it, where one is."""

    name: str = Field(
        description=(
            "What it's for, in a few words, which names it and its branch: "
            "`measure the gate`, `fix-tabs`."
        )
    )
    brief: str = Field(
        description=(
            "Its brief, whole: what to do, what you know already, and what to "
            "say back. It's your message to it, its first turn."
        )
    )
    concern: str | None = Field(
        default=None,
        description=(
            "A concern to hand over with it: an existing one's name or number, "
            "or a new title, which opens one under the one you work in. It "
            "holds it and works in it, its reports land there, and the "
            "operator talks with it directly: once handed, it's theirs. Left "
            "out, it's a side job for you, its report coming back to you."
        ),
    )
    start: str | None = Field(
        default=None,
        description=(
            "The local branch its worktree starts from, by its short name. Left "
            "out, the branch work starts from, as the declaration names it."
        ),
    )
    repository: str | None = Field(
        default=None,
        description=(
            "Another repository to start it in, by the name your guidance lists "
            "it under: refused yet, since no room of yours can make its worktree "
            "there; `concern` with `repository` files what you'd have done there. "
            "Left out, this repository."
        ),
    )


class Made(Model):
    """A worktree made for a spawn: of which repository, where, on which branch, at which commit."""

    repository: Path
    """The repository's common git directory."""
    root: Path
    branch: str
    commit: str


class SpawnError(InboxError):
    """A spawn that can't be made, and why."""


class SpawnedSession(Model):
    """What the host keeps of a session an agent spawned, by its session's id (`Layout.spawned_session`): what its launcher was started with, the commit its worktree started at, and why its worktree stayed once its spawner's launch was gone, where a side job's did."""

    spec: HeldLaunch
    commit: str
    """The commit its worktree was made at, where its branch started."""
    stays: str | None = None
    """What a side job's worktree held, which stays once its spawner's launch
    is gone, as the operator was told once (`SpawnTools.left`); none before
    they were."""
    launches: list[str] = []
    """Each launch the host started for the session, its spawn's and each
    relaunch's, by id: the operator's words restart it only where its last
    words were heard in one of them (`SpawnTools.restarted`)."""


class SpawnCall(Model, arbitrary_types_allowed=True):
    """One conversation of a launch spawning a session: the tool's function, bound to it."""

    tools: SpawnTools
    launch: Register
    party: Conversation
    call: str | None = None

    async def spawn(self, spawn: Spawn) -> str:
        """Start an agent in a room and worktree of its own.

        A session of its own, as yours is, held by lup with no terminal, in a
        worktree lup makes on a branch named after `name`, from the branch
        `start` names or the one work starts from, its environment made as
        its room starts. Your brief is its first turn. The call returns at
        once, while its room starts.

        Two kinds, by whether you hand it a concern:
        - **A side job for you** (no `concern`), such as a search or a
          measurement: its report, what it ends its turn saying, reaches you
          whole, wakes you, and returns your `wait_inbox`, and it ends there.
          `send` reaches it by its name, resuming it with all it knew, its
          answer coming back the same way.
        - **A concern handed over** (`concern`): it holds it and works in it,
          what it says lands there, the operator talks with it directly, and
          it lands its own branch. You hear, in a line, when it says the
          concern is done and when it ends, not its reports, which you read in
          the concern. Once handed, it's theirs.

        The runtime is yours. How many run at once has a high limit, against
        loops of spawns.
        """
        try:
            return await self.tools.spawned(self, spawn)
        except InboxError as refused:
            if refused.__cause__ is not None:
                # What the host's git said stays with the operator: it can quote
                # a file a link the room planted leads to.
                self.tools.delivery.say(
                    f"{self.launch.launch}: a spawn was refused, {refused}: "
                    f"{refused.__cause__}"
                )
            return f"Not spawned: {refused}"

    def tools_bound(self) -> list[AgentTool]:
        """Return the tool, bound to this conversation."""
        return [lup.tool(self.spawn)]


class Reserved(Model):
    """A spawn under way: its repository's git directory, the branch it took, once it took one, and the session it resumes, where it resumes one."""

    token: str
    repository: Path
    branch: str | None = None
    session: str | None = None
    """The runtime's id for the session it resumes, so no two launches resume
    one at once."""


class SpawnTools(MutableModel, RoomTools, arbitrary_types_allowed=True):
    """The `spawn` tool, as the host service serves it to each launched session; what's under way changes as calls come and go."""

    delivery: Delivery
    layout: Layout
    spawning: Spawning
    """Starts a launcher apart from the service, as `lup-dev spawn` does."""
    launchers: Launchers
    served: Callable[[], list[Register]]
    """The launches the service serves, as their records stand."""
    mint: Callable[[str], str | None]
    """Mints a new session's id for the runtime named; none where it has no
    session held with no terminal."""
    starting: Callable[[Path], str]
    """The branch work starts from in the repository at the git directory
    given, as the declaration approved names it."""
    run: Callable[[Program], Awaitable[Ran]] = executed
    """Runs one program on the host: `podman exec` into the spawner's room,
    killed past its `within` with what it started, which ends with status
    124."""
    limit: int = 32
    """How many sessions agents spawned may run at once on the host, every
    repository's: against a loop of spawns, each about 300 MB of a held runtime
    while it runs."""
    each: int = 16
    """How many of those may be one repository's, so one room's loop leaves
    the other repositories theirs."""
    longest: int = 48
    """How long a branch's name made from a spawn's `name` may be, so a number
    added where it's taken keeps it within a launch's name (`handle`)."""
    numbers: int = 100
    """How many numbers a branch's name takes, `-2` to this, before the spawn
    is refused: a room can make as many `tree/<name>-<n>` as it likes."""
    reading: timedelta = timedelta(seconds=30)
    """How long git on the host may take to read refs, before it's killed, as
    long as `Git.limit` gives any: a room can make a ref a pipe nothing
    writes."""
    making: timedelta = timedelta(minutes=2)
    """How long the spawner's room's git may take to make the worktree, a
    checkout of the whole tree, before `podman exec` is killed with what it
    started and the spawn refused."""
    heard: int = 16 * 1024
    """The most of what the room's git prints that's kept, its end, which a
    refusal quotes: the room chooses that git, which could print without end
    into the service's memory (`Program.kept`)."""
    killing: int = 3
    """How many of a launch's commands in its room, making or removing a
    spawn's worktree, may be running or killed past `making` at once, those
    killed counted for good: killing `podman exec` can leave the room's git
    running, a room able to make it wait for good, and with it a process of
    podman's on the host (conmon), until the room stops. Each running may be
    killed, so what a launch leaves so is `killing` at most; past it, a command
    isn't run, and once that many were killed, its spawns are refused. Another
    room of the repository can make this one's git wait too, through the refs
    they share, using up its count: bounded, and refusing is the safe way to
    err."""
    killed: dict[str, int] = {}
    """By launch, how many of its commands in its room were killed past
    `making`: kept while the service runs."""
    running: dict[str, int] = {}
    """By launch, how many of its commands in its room are running."""
    reserved: list[Reserved] = []
    """Each spawn under way, from its count against the limits to its
    launcher's start, with the branch it took once it took one: counted, so
    two calls at once can't both pass a limit, and no two take one branch.
    Nothing is held across what a room can slow."""

    @override
    def specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in SpawnCall.model_construct().tools_bound()]

    @override
    async def call(
        self,
        name: str,
        arguments: JsonValue,
        launch: Register,
        caller: Caller,
        call: str | None,
    ) -> Answer:
        """Run `spawn` for `caller`, a conversation of `launch`; arguments that don't validate are refused, saying why."""
        party = Conversation(
            repository=launch.repository,
            runtime=launch.runtime,
            session=caller.session,
            agent=caller.agent,
        )
        bound = SpawnCall(tools=self, launch=launch, party=party, call=call)
        [tool] = [tool for tool in bound.tools_bound() if tool.spec().name == name]
        match await tool.call(arguments):
            case Accepted(answer=answer):
                return Answer(text=str(answer))
            case Rejected(rejection=rejection):
                return Answer(text=f"Refused: {rejection.reason}", error=True)

    async def spawned(self, bound: SpawnCall, spawn: Spawn) -> str:
        """Start the session `spawn` asks for, of `bound`'s conversation, and say what it is; raise `SpawnError` where it can't be, having made nothing where that's known before.

        Everything the call names is read first: the runtime, the name, the
        limit, what it starts from, the concern. Then the worktree is made in
        the spawner's room, the concern opened where it's a new title, the
        launcher started, the concern handed, and the brief sent, the spawn
        counted against the limits from the first await until its launcher has
        started.
        """
        launch = bound.launch
        session = self.mint(launch.runtime)
        if session is None:
            message = (
                f"{launch.runtime} has no session lup holds with no terminal yet, so "
                "a session of yours spawns none"
            )
            raise SpawnError(message)
        wanted = branch_named(spawn.name)
        if not wanted:
            message = f"`{spawn.name}` holds no letter or digit to name it by"
            raise SpawnError(message)
        if len(wanted) > self.longest:
            message = (
                f"`{spawn.name}` is long for a name: a few words, its branch "
                f"`{wanted}` {self.longest} characters at most"
            )
            raise SpawnError(message)
        if not spawn.brief.strip():
            message = "its brief is empty: it would start with nothing to do"
            raise SpawnError(message)
        capped = self.capped(launch)
        if capped is not None:
            raise SpawnError(capped)
        repository = self.target(launch, spawn.repository)
        if repository != launch.repository:
            # lup: defer(issue=117, why="a session spawned in another repository needs its worktree made by a room of that repository walled as its sessions' rooms are, which lup doesn't start yet: the spawner's room can't reach it, and another session's room isn't the spawner's to run git in")
            message = (
                f"your project lets its sessions spawn in `{spawn.repository}`, but "
                "a session there needs its worktree made by a room of that "
                "repository's, which lup doesn't start yet: file what you'd have "
                "done there with `concern`, given that `repository`"
            )
            raise SpawnError(message)
        reserved = self.limited(repository)
        try:
            return await self.started(bound, spawn, wanted, session, reserved)
        finally:
            self.reserved = [
                each for each in self.reserved if each.token != reserved.token
            ]

    async def started(
        self,
        bound: SpawnCall,
        spawn: Spawn,
        wanted: str,
        session: str,
        reserved: Reserved,
    ) -> str:
        """Start the session `spawn` asks for, of `bound`'s conversation, counted as `reserved` in the repository it starts in, its branch named `wanted` or the next number free, its session `session`; raise `SpawnError` where it can't be."""
        launch, party = bound.launch, bound.party
        repository = reserved.repository
        elsewhere = repository != launch.repository
        git = Git.pinned(repository)
        start = spawn.start or self.starting(repository)
        commit = await started_at(git, start, self.reading)
        concerns = self.delivery.concerns()
        if spawn.concern is not None:
            found = concerns.found(repository, spawn.concern)
            numbered = spawn.concern.strip().removeprefix("@").isdigit()
            whose = "that repository" if elsewhere else "this repository"
            if (found is None and numbered) or (
                found is not None and not isinstance(found, Thread)
            ):
                message = f"`{spawn.concern}` names no concern of {whose}"
                raise SpawnError(message)
            if (
                elsewhere
                and isinstance(found, Thread)
                and concerns.answering(found) is not None
            ):
                message = (
                    f"@{found.name} is answered for by someone running in that "
                    "repository, whose it stays: a session spawned from another "
                    "repository takes a concern nobody running there answers for, "
                    "or opens one with a new title"
                )
                raise SpawnError(message)
        taken = await branches(git, wanted, self.reading)
        made = await self.worktree(
            launch, repository, self.claimed(reserved, wanted, taken), commit
        )
        try:
            thread = self.handing(party, repository, spawn.concern)
        except InboxError as unopened:
            message = f"{unopened}{await self.unmade(launch, made)}"
            raise SpawnError(message) from unopened
        child = Conversation(
            repository=repository, runtime=launch.runtime, session=session
        )
        spawner = Spawner(
            launch=launch.launch,
            conversation=party,
            concern=None if thread is None else thread.number,
        )

        def spawned_as(minted: str) -> HeldLaunch:
            return HeldLaunch(
                launch=minted,
                runtime=launch.runtime,
                worktree=made.root,
                session=session,
                repository=repository,
                spawner=spawner,
                arguments=modelled(launch.model),
            )

        try:
            starting = await self.spawning.started(spawned_as)
        except (ProgramError, OSError) as unstarted:
            message = (
                f"its launcher didn't start: {unstarted}"
                f"{await self.unmade(launch, made)}{opened(thread)}"
            )
            raise SpawnError(message) from unstarted
        try:
            handed = None if thread is None else concerns.handed(party, thread, child)
            brief = Message(kind="note", text=spawn.brief)
            sent = await self.delivery.posted(
                brief, party, child, launch.worktree, call=bound.call
            )
            # The ledger keeps an item in its sender's repository, and its
            # threads in theirs: the operator's copy of a brief to another
            # repository's session lands where its spawner works.
            await self.delivery.copied(
                brief,
                child,
                party,
                launch.worktree,
                call=bound.call,
                thread=None if elsewhere else handed,
            )
        except (InboxError, OSError) as unsent:
            # A session up with no brief would wait on nobody: its launcher is
            # ended before it starts one.
            launcher = self.launchers.started.get(starting.launch)
            if launcher is not None:
                await asyncio.to_thread(launcher.end, self.making)
            message = (
                f"its brief wasn't sent, so its launcher was ended: {unsent}"
                f"{await self.unmade(launch, made)}{opened(thread)}"
            )
            raise SpawnError(message) from unsent
        write_model(
            self.layout.spawned_session(session),
            SpawnedSession(
                spec=spawned_as(starting.launch),
                commit=made.commit,
                launches=[starting.launch],
            ),
        )
        return said(made, start, sent.number, starting.log, handed)

    def handing(
        self, party: Conversation, repository: Path, said: str | None
    ) -> Thread | None:
        """Return the concern `party`'s spawn in `repository` hands over, where it names one, opening it where it's a new title: in their own repository, an existing one, or one under where they work (`Concerns.addressed`); in another, an existing one of it, or one at its top, which nobody holds until it's handed (`Concerns.opened`), so no session of theirs holds what's another repository's even for a moment."""
        if said is None:
            return None
        concerns = self.delivery.concerns()
        if repository == party.repository:
            return concerns.addressed(party, said)
        found = concerns.found(repository, said)
        if isinstance(found, Thread):
            return found
        return concerns.opened(party, said.strip(), repository, taken=True)

    def target(self, launch: Register, named: str | None) -> Path:
        """Return the common git directory of the repository a spawn of `launch` starts in: `launch`'s own where `named` is none; else the one its approved declaration lets its sessions spawn in by that name (`spawnable_in`)."""
        if named is None:
            return launch.repository
        return spawnable_in(self.layout, launch.repository, named)

    async def resumed(
        self, ended: Conversation, spawner: Conversation, launch: Register
    ) -> str:
        """Resume `ended`, a session `spawner`, of `launch`, spawned, that has ended, a side job at its report or a builder the operator ended, its session held again in its worktree with its context, as `lup-dev spawn --resume` holds one, the message they just sent it waiting for its call-up; say how it went, after `send`'s answer, or nothing where it runs, or is none of theirs.

        Its spawner is recorded again as the conversation that sent, of the
        launch they send from, the concern a builder was handed kept.
        """
        spec = self.kept(ended)
        if spec is None or spec.spawner is None:
            return ""
        if spec.spawner.conversation != spawner:
            return ""
        how = (
            "It ended at its report" if spec.spawner.concern is None else "It had ended"
        )
        again = spec.spawner.model_copy(update={"launch": launch.launch})
        try:
            starting = await self.relaunched(
                spec, ended, again, modelled(launch.model), launch
            )
        except SpawnError as refused:
            return f" {how}, and isn't resumed: {refused}."
        if starting is None:
            return ""
        return (
            f" {how}: it's resumed held in {spec.worktree}, with all it knew, your "
            f"message its first turn once it's up; its launcher writes to "
            f"{starting.log}."
        )

    async def restarted(self, ended: Conversation) -> str | None:
        """Restart `ended`, a session an agent spawned that has ended, its session held again in its worktree with its context, as `lup-dev spawn --resume` holds one, where the operator's words reached it, which wait for its call-up; say why not where it can't be, and nothing where it is, it runs, or no agent spawned it.

        It answers to the spawner it was spawned by, as recorded then, the
        concern a builder was handed kept, and runs on the arguments it was
        spawned with; marked restarted (`Spawner.restarted`), so a side job
        runs to its next report though its spawner's launch has ended.
        """
        record = self.record(ended)
        spawner = None if record is None else record.spec.spawner
        if record is None or spawner is None:
            return None
        spec = record.spec
        place = self.delivery.inbox.whereabouts.get(ended.key())
        if place is None or place.launch not in record.launches:
            # A room of a terminal launch can name an ended session's id: the
            # words the operator answered may be that room's, not its.
            return (
                "its last words were heard in a launch lup didn't start for it, so "
                "they may be another session's, speaking as it"
            )
        try:
            await self.relaunched(
                spec,
                ended,
                spawner.model_copy(update={"restarted": True}),
                spec.arguments,
                None,
            )
        except SpawnError as refused:
            return str(refused)
        return None

    def kept(self, ended: Conversation) -> HeldLaunch | None:
        """Return what `ended`, a session's own conversation, was started with where an agent spawned it (`record`)."""
        record = self.record(ended)
        return None if record is None else record.spec

    def record(self, ended: Conversation) -> SpawnedSession | None:
        """Return what the host keeps of `ended`, a session's own conversation, where an agent spawned it (`Layout.spawned_session`), by the session's id lup minted, so nothing a room says names its worktree: of its repository and runtime alone, so a room naming that id from another repository reaches none of it. None for a subagent, or where nothing's kept or it doesn't read."""
        if ended.agent is not None:
            return None
        try:
            kept = self.layout.spawned_session(segment(ended.session))
            record = SpawnedSession.model_validate_json(kept.read_text())
        except ValueError, OSError:
            return None
        spec = record.spec
        if (spec.repository, spec.runtime) != (ended.repository, ended.runtime):
            return None
        return record

    async def relaunched(
        self,
        spec: HeldLaunch,
        ended: Conversation,
        spawner: Spawner,
        arguments: list[str],
        asking: Register | None,
    ) -> Starting | None:
        """Start a launcher resuming `ended`, spawned as `spec` says, held in its worktree, answering to `spawner`, on `arguments`, counted against the limits as a spawn is, and return it starting; none where a launch runs it, or one is starting on it. Raise `SpawnError` where it can't be resumed, saying why.

        Whether a launch runs it or starts on it, its repository's session of
        that id, and its count, are read and taken with nothing awaited
        between, so two at once start one. Where its worktree is gone, it's
        made again at the same path (`remade`), where its runtime finds its
        record, in the room of `asking`, its spawner's launch, where they ask.
        """
        session = ended.session
        repository = spec.repository or ended.repository
        starting = [
            each.resume or each.session
            for each in self.launchers.spawning(self.layout)
            if (each.repository or repository) == repository
        ]
        if (
            any(
                each.session == session and each.repository == repository
                for each in self.served()
            )
            or session in starting
            or any(
                each.session == session and each.repository == repository
                for each in self.reserved
            )
        ):
            return None
        reserved = self.limited(repository)
        self.reserved = [
            each.model_copy(update={"session": session})
            if each.token == reserved.token
            else each
            for each in self.reserved
        ]
        try:
            if not spec.worktree.is_dir(follow_symlinks=False):
                await self.remade(repository, spec.worktree, asking)
            starting = await self.spawning.started(
                lambda minted: spec.model_copy(
                    update={
                        "launch": minted,
                        "session": None,
                        "resume": session,
                        "spawner": spawner,
                        "arguments": arguments,
                    }
                )
            )
        except (ProgramError, OSError) as unstarted:
            message = f"its launcher didn't start: {unstarted}"
            raise SpawnError(message) from unstarted
        finally:
            self.reserved = [
                each for each in self.reserved if each.token != reserved.token
            ]
        record = self.record(ended)
        if record is not None:
            write_model(
                self.layout.spawned_session(segment(session)),
                record.model_copy(
                    update={"launches": [*record.launches, starting.launch]}
                ),
            )
        return starting

    async def left(self) -> None:
        """Tell the operator once of each side job's worktree left once the launch that spawned it is gone, nothing running it: whether its branch holds anything beyond where it started or what's on the integration branch or the running one, as git pinned to the repository reads their refs, in its own threads. lup removes none (`SpawnedSession.stays`). The sweep runs it, apart (`Sweeping.left`).

        Removing one would need git in a room of the repository that's no
        session's own, its spawner's room and its own both gone: the host's
        git runs the filters a room's commits pick and follows the links it
        planted, and another session's room isn't theirs to run git in.
        Whether it's merged is read with refs alone; a look that fails is
        passed over, said in the service's log, for the next sweep.
        """
        served = self.served()
        launches = [each.launch for each in served]
        running = [(each.repository, each.session) for each in served]
        records = await asyncio.to_thread(self.recorded)
        for kept in records:
            spec = kept.spec
            spawner = spec.spawner
            if (
                spawner is None
                or spawner.concern is not None
                or kept.stays is not None
                or spawner.launch in launches
                or spec.session is None
                or spec.repository is None
                or (spec.repository, spec.session) in running
            ):
                continue
            if not await asyncio.to_thread(spec.worktree.is_dir, follow_symlinks=False):
                continue
            git = Git.pinned(spec.repository)
            try:
                unmerged = await self.unmerged(
                    git, spec.repository, spec.worktree.name, kept.commit
                )
            except GitError as failed:
                self.delivery.say(
                    f"The worktree of side job {spec.session}, {spec.worktree}, wasn't "
                    f"looked at: {failed}"
                )
                continue
            branches = approved_branches(self.layout, spec.repository)
            stays = (
                f"its branch `{spec.worktree.name}` holds no commit beyond where it "
                f"started or what's on `{branches.integration}` or "
                f"`{branches.running}`, or is gone; what's uncommitted in it, lup "
                "doesn't look at"
                if unmerged is None
                else unmerged
            )
            write_model(
                self.layout.spawned_session(segment(spec.session)),
                kept.model_copy(update={"stays": stays}),
            )
            await self.told_stays(spec, stays)

    def recorded(self) -> list[SpawnedSession]:
        """Return each spawned session's record the host keeps; one that doesn't read passed over."""

        def read() -> Iterator[SpawnedSession]:
            try:
                paths = sorted(self.layout.spawned_sessions.glob("*.json"))
            except OSError:
                return
            for path in paths:
                try:
                    yield SpawnedSession.model_validate_json(path.read_text())
                except ValueError, OSError:
                    continue

        return list(read())

    async def unmerged(
        self, git: Git, repository: Path, branch: str, commit: str
    ) -> str | None:
        """Say what of `branch`, started at `commit`, is on neither the integration branch nor the running one, as git pinned to `repository` reads their refs, the running one where the host computes it as it computed it (`running_tip`); none where it's all there, or the branch is gone. Raise `GitError` where git can't read them: none would take a failure for nothing to keep."""
        found = await threaded(
            repository,
            git.run,
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"refs/heads/{branch}^{{commit}}",
            ok=[0, 1],
            within=self.reading,
        )
        tip = found.output.decode().strip()
        if found.status != 0 or not tip or tip == commit:
            return None
        branches = approved_branches(self.layout, repository)
        running = running_tip(self.layout, repository)
        heads = [
            f"refs/heads/{branches.integration}",
            running or f"refs/heads/{branches.running}",
        ]
        for head in heads:
            merged = await threaded(
                repository,
                git.run,
                "merge-base",
                "--is-ancestor",
                "--end-of-options",
                tip,
                head,
                ok=[0, 1],
                within=self.reading,
            )
            if merged.status == 0:
                return None
        return (
            f"its branch `{branch}` holds commits on neither "
            f"`{branches.integration}` nor `{branches.running}`"
        )

    async def told_stays(self, spec: HeldLaunch, stays: str) -> None:
        """Tell the operator once that a side job's worktree stays, though the launch that spawned it is gone, and what it holds: a note from it, landing where it worked."""
        session = spec.session or spec.resume
        if session is None or spec.repository is None:
            return
        side = Conversation(
            repository=spec.repository, runtime=spec.runtime, session=session
        )
        who = self.delivery.roster().who(side)
        text = (
            f"lup: {who}'s worktree, {spec.worktree}, stays, though the launch that "
            f"spawned it has ended: {stays}. lup leaves it, and its branch, for you "
            "to keep or remove."
        )
        await self.delivery.posted(
            Message(kind="note", text=text), side, Operator(), None
        )

    async def remade(
        self, repository: Path, root: Path, launch: Register | None
    ) -> None:
        """Make a spawned session's worktree, gone, again at `root`, where it was in `repository`, by the repository's own git in the room of `launch`, its spawner's asking (`git_in`): on its branch, named as its worktree is, where that's still there, else on a new one of that name from the branch work starts from, its work having landed. Raise `SpawnError` where it can't be, saying why: where no spawner of the repository asks, as where the operator's words restart it, no room is there that's theirs to run git in."""
        branch = root.name
        if root != made_root(repository, branch):
            message = f"its worktree, {root}, is gone, and wasn't one lup made"
            raise SpawnError(message)
        if launch is None or launch.repository != repository:
            message = (
                f"its worktree, {root}, is gone, and only its spawner's room makes it "
                "again: their `send` brings it back"
            )
            raise SpawnError(message)
        git = Git.pinned(repository)
        try:
            await started_at(git, branch, self.reading)
            adding = [str(root), branch]
        except SpawnError:
            start = self.starting(repository)
            commit = await started_at(git, start, self.reading)
            adding = ["-b", branch, str(root), commit]
        ran = await self.git_in(
            launch, repository, ["worktree", "add", "--quiet", *adding]
        )
        if ran.status != 0:
            said = ran.said() or f"it ended with status {ran.status}"
            message = f"its worktree, {root}, is gone, and wasn't made again: {said}"
            raise SpawnError(message)

    async def unmade(self, launch: Register, made: Made) -> str:
        """Remove a spawn's worktree and branch, which nothing ran in yet, by its repository's own git in a room (`git_in`), stopped past `making`, and say how it went, to follow a refusal's reason."""
        stay = f"; its worktree, {made.root}, and branch `{made.branch}` stay"
        steps = [
            ["worktree", "remove", "--force", str(made.root)],
            ["branch", "-D", made.branch],
        ]
        for step in steps:
            try:
                ran = await self.git_in(launch, made.repository, step)
            except SpawnError:
                # Another command run here could be killed too, leaving one more.
                return stay
            if ran.status != 0:
                return stay
        return "; its worktree and branch were removed"

    async def git_in(
        self, launch: Register, repository: Path, arguments: list[str]
    ) -> Ran:
        """Run git with `arguments` on `repository`, `launch`'s own, in `launch`'s room, the one asking, through `podman exec`, counted against its cap (`in_room`), and never on the host: what it does there, that room could do itself. Never in another session's room, whose own tools, login and lent directories what a room planted in the shared git directory would reach there; never in a room started for it, which wouldn't hide what a session's room hides from it.

        Pinned to the git directory (`pinning`), its hooks and file monitor
        off, so neither a `commondir` nor a `config` the room planted there
        leads its git to run anything of the room's. Return how it ended,
        whatever its status, 124 where it ran past `making`; raise `SpawnError`
        where `launch` isn't the repository's, or names no room.
        """
        container = launch.container
        if repository != launch.repository or container is None:
            message = "your launch names no room of that repository to run its git in"
            raise SpawnError(message)
        program = Program(
            command=[
                "podman",
                "exec",
                "--workdir",
                str(repository),
                container,
                *pinning(repository),
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                *arguments,
            ],
            payload="",
            directory=repository,
            within=self.making,
            kept=self.heard,
        )
        return await self.in_room(launch, program)

    def limited(self, repository: Path) -> Reserved:
        """Count a spawn in `repository` against the limits, the host's and the repository's, and return it counted; refuse it past either, naming those that run.

        Those counted are the launches served that an agent spawned, those
        whose launcher started and hasn't registered, and the spawns under way.
        Read and counted at once, with nothing awaited between, so two calls
        can't both pass.
        """
        running = {
            each.launch: (each.repository, each.name or each.launch)
            for each in self.served()
            if each.spawner is not None
        }
        starting = {
            each.launch: (each.repository or repository, each.worktree.name)
            for each in self.launchers.spawning(self.layout)
            if each.launch not in running
        }
        under_way = {
            each.token: (each.repository, each.branch or "a spawn under way")
            for each in self.reserved
        }
        counted = {**running, **starting, **under_way}
        theirs = {
            key: name for key, (where, name) in counted.items() if where == repository
        }
        for limit, among, whose in [
            (self.limit, {key: name for key, (_, name) in counted.items()}, ""),
            (self.each, theirs, " of this repository"),
        ]:
            if len(among) >= limit:
                names = ", ".join(f"`{each}`" for each in sorted(among.values()))
                runs = (
                    f"session agents spawned{whose} runs"
                    if len(among) == 1
                    else f"sessions agents spawned{whose} run"
                )
                message = (
                    f"{len(among)} {runs} already, the most that may at once, "
                    f"against loops of spawns: {names}. One ends with `lup-dev "
                    "spawn end`, which is the operator's"
                )
                raise SpawnError(message)
        reserved = Reserved(token=secrets.token_hex(8), repository=repository)
        self.reserved = [*self.reserved, reserved]
        return reserved

    def claimed(self, reserved: Reserved, wanted: str, taken: list[str]) -> str:
        """Take for `reserved` the first of `wanted`, `wanted-2`, `wanted-3` up to `numbers` that's no branch `taken` names, no path in the repository's `tree/`, no other spawn's under way, and no worktree a session spawned before worked in, which the host keeps for good (`Layout.spawned_session`), so a restart of that one never starts in another's; refuse where none is.

        Read and taken at once, with nothing awaited between.
        """
        others = [
            each.branch
            for each in self.reserved
            if each.repository == reserved.repository and each.token != reserved.token
        ]
        before = [each.spec.worktree for each in self.recorded()]
        candidates = [wanted, *(f"{wanted}-{n}" for n in range(2, self.numbers + 1))]
        free = next(
            (
                each
                for each in candidates
                if each not in taken
                and each not in others
                and made_root(reserved.repository, each) not in before
                and not made_root(reserved.repository, each).exists(
                    follow_symlinks=False
                )
            ),
            None,
        )
        if free is None:
            message = (
                f"`{wanted}` and `{wanted}-2` to `{wanted}-{self.numbers}` are each a "
                "branch or a worktree already: name it otherwise"
            )
            raise SpawnError(message)
        self.reserved = [
            each.model_copy(update={"branch": free})
            if each.token == reserved.token
            else each
            for each in self.reserved
        ]
        return free

    async def worktree(
        self, launch: Register, repository: Path, branch: str, commit: str
    ) -> Made:
        """Make the spawn's worktree at `tree/<branch>` in `repository`'s git directory, on the branch `branch`, at `commit`, by the repository's own git in a room (`git_in`), stopped past `making`; raise `SpawnError` where no room can run it, or git fails or runs past it, saying why."""
        root = made_root(repository, branch)
        made = Made(repository=repository, root=root, branch=branch, commit=commit)
        where = "your room"
        ran = await self.git_in(
            launch,
            repository,
            ["worktree", "add", "--quiet", "-b", branch, str(root), commit],
        )
        if ran.status == 124:  # `executed` killed it past `within`
            # Killing `podman exec` may leave the room's git running, which
            # can make the worktree after all: it's removed all the same.
            message = (
                f"its worktree, {root}, wasn't made in {where} within "
                f"{self.making}, so its git was stopped"
                f"{await self.unmade(launch, made)}"
            )
            raise SpawnError(message)
        if ran.status != 0:
            said = ran.said() or f"it ended with status {ran.status}"
            message = f"its worktree, {root}, wasn't made in {where}: {said}"
            raise SpawnError(message)
        return made

    def capped(self, launch: Register) -> str | None:
        """Say why `launch` runs no more commands in its room for a spawn, where `killing` of them run or were killed past `making`; none where fewer do."""
        killed = self.killed.get(launch.launch, 0)
        running = self.running.get(launch.launch, 0)
        if killed + running < self.killing:
            return None
        if killed >= self.killing:
            return (
                f"{killed} commands lup ran in your room to make or remove a "
                f"spawn's worktree ran past {self.making} and were killed, each of "
                "which can leave your room's git running, and a process of "
                "podman's on the host with it, so your launch spawns no more: a "
                "launch started afresh does"
            )
        return (
            f"{running} commands lup runs in your room for spawns are under way, "
            f"and {killed} ran past {self.making} and were killed: each can be "
            "killed, leaving your room's git running, and a process of podman's "
            f"on the host with it, so no more than {self.killing} run or were "
            "killed at once; spawn again once one ends"
        )

    async def in_room(self, launch: Register, program: Program) -> Ran:
        """Run `program`, a command of `launch`'s in its room, counted as running until it ends, and among those killed where `executed` killed it past `making`, its status 124; raise `SpawnError` where it isn't run, `killing` of them running or killed (`capped`).

        The count is read and taken with nothing awaited between, so spawns
        under way at once run no more than it lets.
        """
        capped = self.capped(launch)
        if capped is not None:
            raise SpawnError(capped)
        key = launch.launch
        self.running = {**self.running, key: self.running.get(key, 0) + 1}
        try:
            ran = await self.run(program)
        finally:
            self.running = {**self.running, key: self.running.get(key, 1) - 1}
        if ran.status == 124:
            self.killed = {**self.killed, key: self.killed.get(key, 0) + 1}
        return ran

    async def lost(self, spec: HeldLaunch) -> None:
        """Tell the agent who spawned `spec`'s launch that its session never came up, naming its launcher's log, as a problem from it: loud, so a spawner waiting on its report hears instead. A session resumed so is the one it resumes."""
        spawner = spec.spawner
        session = spec.session or spec.resume
        if spawner is None or session is None:
            return
        log = self.layout.launch(spec.launch).log
        child = Conversation(
            repository=spawner.conversation.repository,
            runtime=spec.runtime,
            session=session,
        )
        # Not what the log says: it can name what's no room's to read, as
        # another repository's paths.
        text = (
            f"lup: the session spawned in {spec.worktree} never came up, so it "
            f"isn't running. Its launcher's log, the operator's to read, is {log}."
        )
        await self.delivery.posted(
            Message(kind="problem", text=text),
            child,
            self.delivery.reaching(spawner.conversation),
            None,
        )


def spawnable_in(layout: Layout, repository: Path, named: str) -> Path:
    """Return the common git directory of the repository `repository`'s approved declaration lets its sessions spawn in, and file concerns into, by the name `named`, as the host kept it when the operator approved it (`Approval.spawnable`), never as a room's copy of the declaration says. Raise `SpawnError` where it names none, isn't on this machine, is `repository` itself, or has nothing approved, which nothing launches in.

    Its git directory is found as a launch finds a related repository's
    (`git_directory`): a clone's `.git` directory or a bare repository's own,
    without running git there, which would follow a `.git` file its rooms
    write.
    """
    approval = approved(layout, repository)
    allowed = [] if approval is None else approval.spawnable
    found = next((each for each in allowed if each.name == named), None)
    if found is None:
        names = ", ".join(f"`{each.name}`" for each in allowed) or "none"
        message = (
            f"`{named}` isn't a repository your project lets its sessions spawn "
            f"in, as the operator approved it: those are {names}"
        )
        raise SpawnError(message)
    other = git_directory(found.path)
    if other is None:
        message = f"`{named}` isn't a repository at {found.path} on this machine"
        raise SpawnError(message)
    if other == repository:
        message = f"`{named}` is this repository: leave `repository` out"
        raise SpawnError(message)
    if approved(layout, other) is None:
        message = (
            f"nothing is approved for `{named}`, so nothing launches there: the "
            "operator launches there once first"
        )
        raise SpawnError(message)
    return other


async def started_at(git: Git, start: str, within: timedelta) -> str:
    """Return the commit the local branch `start` is at, read by git pinned to the repository's common directory, in the repository's own threads (`threaded`), stopped past `within`; raise `SpawnError` where `start` isn't a branch's short name, names no branch there, or git can't read it in time."""
    repository = git.where()
    try:
        branch_name(start)
        await threaded(
            repository,
            git.run,
            "check-ref-format",
            f"refs/heads/{start}",
            within=within,
        )
    except (ValueError, GitError) as unnamed:
        message = f"`{start}` isn't a branch's short name"
        raise SpawnError(message) from unnamed
    try:
        found = await threaded(
            repository,
            git.run,
            "rev-parse",
            "--verify",
            "--quiet",
            "--end-of-options",
            f"refs/heads/{start}^{{commit}}",
            ok=[0, 1],
            within=within,
        )
    except GitError as unread:
        # Not what git said: it reads what links a room planted lead to, any
        # file of the operator's among it, and quotes the line it choked on.
        message = (
            f"`{start}` couldn't be read; what git said is in the host service's log"
        )
        raise SpawnError(message) from unread
    commit = found.output.decode().strip()
    if found.status != 0 or not commit:
        message = f"no local branch `{start}` is there to start from"
        raise SpawnError(message)
    return commit


async def branches(git: Git, wanted: str, within: timedelta) -> list[str]:
    """Return the branches of the repository named `wanted` or `wanted-<anything>`, by their short names, read in one go by git pinned to its common directory, in the repository's own threads (`threaded`), stopped past `within`; raise `SpawnError` where git can't read them in time."""
    try:
        found = await threaded(
            git.where(),
            git.run,
            "for-each-ref",
            "--format=%(refname:lstrip=2)",
            f"refs/heads/{wanted}",
            f"refs/heads/{wanted}-*",
            within=within,
        )
    except GitError as unread:
        # Not what git said, as `started_at` keeps it.
        message = (
            f"the branches named `{wanted}` couldn't be read; what git said is in "
            "the host service's log"
        )
        raise SpawnError(message) from unread
    return found.output.decode().splitlines()


def opened(thread: Thread | None) -> str:
    """Say the concern a refused spawn named stays, where it named one, to follow the refusal's reason."""
    return "" if thread is None else f"; @{thread.name} stays as it is"


def modelled(model: str | None) -> list[str]:
    """Return the runtime's arguments a spawned session runs with: the spawner's launch's model, where its arguments named one, so the agents it spawns run on the model it does, the trial's cheap one among them.

    >>> modelled("haiku"), modelled(None)
    (['--model', 'haiku'], [])
    """
    return [] if model is None else ["--model", model]


def said(made: Made, start: str, brief: int, log: Path, handed: Thread | None) -> str:
    """Say what a spawn started: its name, where it works, which kind it is and where what it says goes."""
    where = (
        f"`{made.branch}` is starting, in {made.root} on its branch `{made.branch}`, "
        f"from `{start}`, at {made.commit}; its brief is @{brief}, its first "
        f"turn, and its launcher writes to {log}."
    )
    if handed is None:
        return (
            f"{where} A side job for you: what it ends its turn saying reaches you "
            "whole, wakes you, and returns your `wait_inbox`, and it ends there. "
            f"`send` to `{made.branch}` resumes it, with all it knew."
        )
    return (
        f"{where} @{handed.name} is theirs now: what it says lands there, where you "
        "can read it, and the operator talks with it directly. You'll hear in a "
        "line when it says it's done, and when it ends."
    )
