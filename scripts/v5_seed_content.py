"""Hand-authored V5 seed content: ten contrast blocks, abstention, off-topic.

Nothing here is copied from ``eval/fresh_eval_2026_10.json`` (the burned set) or
from ``data/vector_val.jsonl`` / ``data/vector_test.jsonl``.  The overlap audit
in ``prepare_v5_dataset.py`` re-checks that claim mechanically.

Each positive entry is ``(query, intent, note)``.  The ``note`` records the
contrast the example is meant to teach, so the pair it belongs to is traceable
in review.  Negatives carry no intent.

Design rule: a block is a *boundary*, so its examples are deliberately close to
one another and differ in the one word (or the presence/absence of a number)
that decides the tool.  A block with only one side populated teaches nothing.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Block 1 -- play vs pause
# ---------------------------------------------------------------------------
# Resume-family -> media_play; hold-family -> media_pause.  Includes negated
# forms ("don't stop") and forms carrying the opposite class's substring
# ("unpause", "un-pause") so the model stops keying on surface tokens.

_B1 = [
    # media_play
    ("play the song", "media_play", "base"),
    ("hit play", "media_play", "terse"),
    ("resume the track", "media_play", "resume"),
    ("unpause the song", "media_play", "negated-pause"),
    ("un-pause the album", "media_play", "negated-pause + hyphen"),
    ("carry on with the album", "media_play", "carry-on"),
    ("keep the music going", "media_play", "keep-going"),
    ("keep playing, don't stop", "media_play", "negated-stop"),
    ("start the music up again", "media_play", "again"),
    ("fire the music back up", "media_play", "fire-up-again"),
    ("put the album back on", "media_play", "back-on"),
    ("get the playlist rolling again", "media_play", "restart"),
    ("let the track keep going", "media_play", "keep-going"),
    ("resume playback from here", "media_play", "resume"),
    ("bring the song back", "media_play", "restore-playback"),
    ("restart the music", "media_play", "restart"),
    ("get playback going again", "media_play", "restart"),
    ("put the song back on", "media_play", "back-on"),
    ("don't pause the album", "media_play", "negated-pause"),
    ("switch the music back on", "media_play", "back-on"),
    ("press play on the track", "media_play", "press-play"),
    ("let the music play on", "media_play", "continue"),
    ("continue the playlist", "media_play", "continue"),
    ("take the song off hold", "media_play", "off-hold"),
    ("resume the album where it was", "media_play", "resume"),

    # media_pause
    ("pause the song", "media_pause", "base"),
    ("hit pause", "media_pause", "terse"),
    ("stop the music", "media_pause", "stop"),
    ("halt the track", "media_pause", "halt"),
    ("suspend playback", "media_pause", "suspend"),
    ("put the song on hold", "media_pause", "hold"),
    ("hold the music for a moment", "media_pause", "hold"),
    ("freeze the track", "media_pause", "freeze"),
    ("freeze the playlist", "media_pause", "freeze"),
    ("kill the track", "media_pause", "kill-class"),
    ("shut the song off", "media_pause", "off"),
    ("don't play the track", "media_pause", "negated-play"),
    ("stop playing that song", "media_pause", "stop"),
    ("put playback on pause", "media_pause", "pause"),
    ("give the music a break", "media_pause", "break"),
    ("pause the album for now", "media_pause", "pause"),
    ("take a break from the track", "media_pause", "break"),
    ("hold the playlist", "media_pause", "hold"),
    ("press pause", "media_pause", "terse"),
    ("stop the audio for a second", "media_pause", "stop"),
    ("leave the song paused", "media_pause", "paused"),
    ("don't let the track play", "media_pause", "negated-play"),
    ("take the track off play", "media_pause", "off"),
    ("stop the playlist where it is", "media_pause", "stop"),
]

# ---------------------------------------------------------------------------
# Block 2 -- mute vs unmute polarity
# ---------------------------------------------------------------------------
# On/off pairs across several carriers (verb, particle, adjective) plus
# colloquial forms.  Includes examples with *no* polarity word at all, which is
# its own decision ("shush", "cut", "hush").

_B2 = [
    # mute
    ("mute the speakers", "mute", "verb"),
    ("mute the sound", "mute", "verb"),
    ("mute the audio", "mute", "verb"),
    ("silence the speakers", "mute", "verb"),
    ("silence the sound", "mute", "verb"),
    ("shush the audio", "mute", "no-polarity-word"),
    ("hush the speakers", "mute", "no-polarity-word"),
    ("cut the sound", "mute", "no-polarity-word"),
    ("kill the audio", "mute", "kill-class"),
    ("turn the sound off", "mute", "particle"),
    ("turn off the sound", "mute", "particle"),
    ("switch the audio off", "mute", "particle"),
    ("disable the sound", "mute", "adjective"),
    ("disable audio", "mute", "adjective"),
    ("shut the sound off", "mute", "particle"),
    ("put the sound off", "mute", "particle"),
    ("set the sound to off", "mute", "state"),
    ("quiet the speakers", "mute", "adjective"),
    ("dampen the audio", "mute", "no-polarity-word"),
    ("silence the computer audio", "mute", "verb"),

    # unmute
    ("unmute the speakers", "unmute", "verb"),
    ("unmute the sound", "unmute", "verb"),
    ("unmute the audio", "unmute", "verb"),
    ("turn the sound back on", "unmute", "particle"),
    ("turn the audio back on", "unmute", "particle"),
    ("bring the sound back", "unmute", "restore"),
    ("bring back the audio", "unmute", "restore"),
    ("restore the sound", "unmute", "restore"),
    ("restore audio", "unmute", "restore"),
    ("re-enable the sound", "unmute", "adjective"),
    ("switch the sound back on", "unmute", "particle"),
    ("switch the audio on", "unmute", "particle"),
    ("turn the sound on", "unmute", "particle"),
    ("get the audio back on", "unmute", "restore"),
    ("put the sound back on", "unmute", "particle"),
    ("sound on please", "unmute", "terse"),
    ("un-silence the speakers", "unmute", "negated-mute"),
    ("cancel the mute", "unmute", "negated-mute"),
    ("undo the silence", "unmute", "negated-mute"),
    ("make the speakers audible again", "unmute", "adjective"),
]

# ---------------------------------------------------------------------------
# Block 3 -- media direction: next vs previous
# ---------------------------------------------------------------------------

_B3 = [
    # media_next
    ("next track", "media_next", "base"),
    ("go to the next song", "media_next", "go-to"),
    ("skip this song", "media_next", "skip"),
    ("skip forward", "media_next", "skip+forward"),
    ("advance to the next track", "media_next", "advance"),
    ("jump ahead one song", "media_next", "ahead"),
    ("play the following song", "media_next", "following"),
    ("move ahead one track", "media_next", "ahead"),
    ("go forward one song", "media_next", "forward"),
    ("take me to the next track", "media_next", "go-to"),
    ("switch to the next song", "media_next", "switch"),
    ("skip to the following track", "media_next", "skip"),
    ("forward one track", "media_next", "terse"),
    ("go on to the next one", "media_next", "go-to"),
    ("press next", "media_next", "terse"),
    ("skip over this track", "media_next", "skip"),
    ("queue the next song", "media_next", "queue"),
    ("skip ahead", "media_next", "skip"),
    ("next song please", "media_next", "terse"),
    ("move on to the next track", "media_next", "go-to"),
    ("play the track after this", "media_next", "following"),
    ("advance the queue", "media_next", "advance"),

    # media_previous
    ("previous track", "media_previous", "base"),
    ("go back one song", "media_previous", "go-back"),
    ("go to the previous song", "media_previous", "go-to"),
    ("play the last track", "media_previous", "last"),
    ("rewind one song", "media_previous", "rewind"),
    ("take me back a track", "media_previous", "back"),
    ("back up a song", "media_previous", "two-sense verb"),
    ("switch to the previous track", "media_previous", "switch"),
    ("return to the earlier song", "media_previous", "earlier"),
    ("go to the song before this", "media_previous", "before"),
    ("previous one please", "media_previous", "terse"),
    ("go backwards one track", "media_previous", "backwards"),
    ("step back one song", "media_previous", "back"),
    ("play the track before this", "media_previous", "before"),
    ("jump back a song", "media_previous", "back"),
    ("back one track", "media_previous", "terse"),
    ("skip backward", "media_previous", "skip"),
    ("play the earlier track", "media_previous", "earlier"),
    ("rewind to the last song", "media_previous", "rewind"),
    ("take me to the previous one", "media_previous", "go-to"),
    ("go to the prior track", "media_previous", "prior"),
    ("previous song", "media_previous", "terse"),
]

# ---------------------------------------------------------------------------
# Block 4 -- battery vs disk vs system stats vs cpu vs ram
# ---------------------------------------------------------------------------
# Five tools that share one family and one carrier verb ("how much ...", "check
# ...", "is ... full").  The carrier is held constant and the noun changes, so
# the boundary is the noun rather than the sentence shape.

_B4 = [
    # get_battery_status
    ("how much battery is left", "get_battery_status", "carrier"),
    ("what's my battery percentage", "get_battery_status", "percentage"),
    ("battery status", "get_battery_status", "base"),
    ("check the battery", "get_battery_status", "carrier"),
    ("how much charge is left", "get_battery_status", "charge"),
    ("is the battery full", "get_battery_status", "is-full"),
    ("battery level", "get_battery_status", "base"),
    ("how long will the battery last", "get_battery_status", "duration"),
    ("am i on battery power", "get_battery_status", "source"),
    ("what percentage is the battery at", "get_battery_status", "percentage"),
    ("do i have much battery left", "get_battery_status", "carrier"),
    ("battery readings", "get_battery_status", "base"),
    ("show the charge level", "get_battery_status", "charge"),

    # get_disk_usage
    ("how much disk space is left", "get_disk_usage", "carrier"),
    ("how much storage is left", "get_disk_usage", "storage"),
    ("check the disk", "get_disk_usage", "carrier"),
    ("disk usage", "get_disk_usage", "base"),
    ("how full is the drive", "get_disk_usage", "is-full"),
    ("free space on the drive", "get_disk_usage", "free"),
    ("storage status", "get_disk_usage", "base"),
    ("how much space do i have", "get_disk_usage", "carrier"),
    ("is the disk full", "get_disk_usage", "is-full"),
    ("remaining disk space", "get_disk_usage", "remaining"),
    ("how many gigabytes are free", "get_disk_usage", "units"),
    ("filesystem usage", "get_disk_usage", "synonym"),
    ("how much room is left on the drive", "get_disk_usage", "room"),
    ("disk capacity", "get_disk_usage", "capacity"),
    ("how much of the drive is used", "get_disk_usage", "used"),

    # get_system_stats
    ("system stats", "get_system_stats", "base"),
    ("show me the system overview", "get_system_stats", "overview"),
    ("give me a summary of the machine", "get_system_stats", "summary"),
    ("overall system information", "get_system_stats", "overview"),
    ("how is the system doing", "get_system_stats", "health"),
    ("machine status", "get_system_stats", "base"),
    ("system health", "get_system_stats", "health"),
    ("show everything about the system", "get_system_stats", "overview"),
    ("overall stats", "get_system_stats", "base"),
    ("system summary", "get_system_stats", "summary"),
    ("give me the full system report", "get_system_stats", "report"),
    ("machine overview", "get_system_stats", "overview"),

    # get_cpu_usage
    ("how hard is the processor working", "get_cpu_usage", "processor"),
    ("cpu usage", "get_cpu_usage", "base"),
    ("processor load", "get_cpu_usage", "processor"),
    ("how busy is the processor", "get_cpu_usage", "processor"),
    ("check the cpu", "get_cpu_usage", "carrier"),
    ("cpu percentage", "get_cpu_usage", "percentage"),
    ("what's the cpu doing", "get_cpu_usage", "activity"),
    ("how much cpu is being used", "get_cpu_usage", "carrier"),
    ("show cpu load", "get_cpu_usage", "load"),
    ("is the processor busy", "get_cpu_usage", "processor"),

    # get_memory_usage
    ("how much memory is in use", "get_memory_usage", "carrier"),
    ("ram usage", "get_memory_usage", "base"),
    ("check my memory", "get_memory_usage", "carrier"),
    ("how much ram is left", "get_memory_usage", "carrier"),
    ("memory status", "get_memory_usage", "base"),
    ("how much memory is free", "get_memory_usage", "free"),
    ("is the ram full", "get_memory_usage", "is-full"),
    ("show ram usage", "get_memory_usage", "load"),
    ("memory consumption", "get_memory_usage", "consumption"),
    ("how much of my memory is used", "get_memory_usage", "used"),
]

# ---------------------------------------------------------------------------
# Block 5 -- indirect battery vocabulary
# ---------------------------------------------------------------------------
# None of these contains the word "battery".  They are the phrasings a person
# uses when they want battery state but do not name it.

_B5 = [
    ("is the charger plugged in", "get_battery_status", "charger"),
    ("is the laptop plugged in", "get_battery_status", "plugged"),
    ("how much juice is left in the battery", "get_battery_status", "juice"),
    ("will the battery die soon", "get_battery_status", "running-out"),
    ("am i about to run out of charge", "get_battery_status", "running-out"),
    ("how much power do i have left", "get_battery_status", "power"),
    ("is the power cord connected", "get_battery_status", "cord"),
    ("am i running on battery", "get_battery_status", "source"),
    ("how long until the battery runs out", "get_battery_status", "duration"),
    ("is it plugged into the wall", "get_battery_status", "source"),
    ("charge remaining", "get_battery_status", "terse"),
    ("what's the charge level", "get_battery_status", "charge"),
    ("is the charger attached", "get_battery_status", "charger"),
    ("am i low on juice", "get_battery_status", "no-storage-sense"),
    ("how much battery life is left", "get_battery_status", "life"),
    ("does the laptop need charging", "get_battery_status", "charging"),
    ("is the power adapter connected", "get_battery_status", "adapter"),
    ("how much energy is left in the battery", "get_battery_status", "energy"),
    ("is my battery about to die", "get_battery_status", "dying"),
    ("what percentage of charge remains", "get_battery_status", "percentage"),
    ("am i plugged in right now", "get_battery_status", "source"),
    ("how much juice does the battery have", "get_battery_status", "juice"),
]

# ---------------------------------------------------------------------------
# Block 5b -- indirect disk/storage vocabulary (same boundary, other side)
# ---------------------------------------------------------------------------

_B4B = [
    ("how many gigs are free", "get_disk_usage", "synonym"),
    ("is there room on the drive", "get_disk_usage", "room"),
    ("how much space is left on my ssd", "get_disk_usage", "hardware"),
    ("is my storage running out", "get_disk_usage", "running-out"),
    ("how many gigabytes are on the disk", "get_disk_usage", "gigabytes"),
    ("do i have space for more files", "get_disk_usage", "space"),
    ("how full is my storage", "get_disk_usage", "is-full"),
    ("am i running out of disk space", "get_disk_usage", "running-out"),
]

# ---------------------------------------------------------------------------
# Block 5b' -- CPU / RAM carried by indirect vocabulary
# ---------------------------------------------------------------------------

_B4_INDIRECT = [
    ("is my processor overloaded", "get_cpu_usage", "processor"),
    ("how loaded is the cpu", "get_cpu_usage", "load"),
    ("is the machine working hard", "get_cpu_usage", "indirect"),
    ("am i running out of memory", "get_memory_usage", "running-out"),
    ("is my ram maxed out", "get_memory_usage", "maxed"),
    ("how much memory is free right now", "get_memory_usage", "carrier"),
]

# ---------------------------------------------------------------------------
# Block 6 -- volume read vs volume change
# ---------------------------------------------------------------------------
# The deciding signal is the presence of a number.  Read queries must map to
# get_volume with *no* argument; write queries must extract the level.  Several
# pairs are identical except for the number.

_B6 = [
    # get_volume -- no number anywhere
    ("what's the volume at", "get_volume", "read"),
    ("what volume is it", "get_volume", "read"),
    ("how loud is it", "get_volume", "read"),
    ("show the volume", "get_volume", "read"),
    ("what is the sound level", "get_volume", "read"),
    ("check the volume", "get_volume", "read"),
    ("is the volume up", "get_volume", "read"),
    ("what's the audio level", "get_volume", "read"),
    ("how loud are the speakers", "get_volume", "read"),
    ("volume status", "get_volume", "read"),
    ("what's my current volume", "get_volume", "read"),
    ("tell me the volume level", "get_volume", "read"),
    ("how high is the sound", "get_volume", "read"),
    ("what level is the audio at", "get_volume", "read"),
    ("is it muted", "get_volume", "read-mutedness"),
    ("is the sound muted", "get_volume", "read-mutedness"),
    ("speaker level", "get_volume", "read"),
    ("what's the speaker volume", "get_volume", "read"),
    ("current sound level", "get_volume", "read"),
    ("what percentage is the volume", "get_volume", "read"),

    # set_volume -- number present
    ("set the volume to 17", "set_volume", "write"),
    ("volume 42", "set_volume", "write"),
    ("turn the volume down to 25", "set_volume", "write-down"),
    ("turn the volume up to 85", "set_volume", "write-up"),
    ("put the volume at 60", "set_volume", "write"),
    ("crank the sound to 70", "set_volume", "write"),
    ("make the volume 33", "set_volume", "write"),
    ("set the sound to 44", "set_volume", "write"),
    ("change the volume to 55", "set_volume", "write"),
    ("bump the volume to 90", "set_volume", "write-up"),
    ("lower the volume to 12", "set_volume", "write-down"),
    ("raise the volume to 76", "set_volume", "write-up"),
    ("set volume 99", "set_volume", "write"),
    ("set the audio to 64", "set_volume", "write"),
    ("set the speakers to 20", "set_volume", "write"),
    ("bring the volume to 50", "set_volume", "write"),
    ("adjust the volume to 38", "set_volume", "write"),
    ("volume to 71", "set_volume", "terse"),
    ("sound at 30", "set_volume", "terse"),
    ("turn the sound to 82", "set_volume", "write"),
    ("set the volume at 47", "set_volume", "write"),
    ("dial the volume to 26", "set_volume", "write"),
]

# ---------------------------------------------------------------------------
# Block 7 -- search files vs launch app (vs close app, vs running apps)
# ---------------------------------------------------------------------------
# Same nouns, different head verb.  "find chrome" searches; "open chrome"
# launches; "close chrome" quits; "what's running" lists.

_B7 = [
    # search_files
    ("find my tax return", "search_files", "find"),
    ("search for the invoice", "search_files", "search"),
    ("look for the meeting agenda", "search_files", "look-for"),
    ("locate my notes", "search_files", "locate"),
    ("where is the budget spreadsheet", "search_files", "where-is"),
    ("find the project proposal", "search_files", "find"),
    ("search my documents for the contract", "search_files", "place+target"),
    ("look for the lease agreement", "search_files", "look-for"),
    ("where did i save the essay", "search_files", "where-did"),
    ("find the photo album", "search_files", "find"),
    ("locate the presentation slides", "search_files", "locate"),
    ("search the drive for invoices", "search_files", "place+target"),
    ("find my resume file", "search_files", "find"),
    ("where are my lecture notes", "search_files", "where-are"),
    ("look for the report draft", "search_files", "look-for"),
    ("find the spreadsheet from last week", "search_files", "find"),
    ("search for the recipe file", "search_files", "search"),
    ("locate the tax documents", "search_files", "locate"),
    ("where's the scan of my passport", "search_files", "where-is"),
    ("find the contract copy", "search_files", "find"),
    ("find chrome", "search_files", "search-vs-launch pair"),
    ("search for notepad", "search_files", "search-vs-launch pair"),
    ("where is firefox", "search_files", "search-vs-launch pair"),
    ("look for the calculator", "search_files", "search-vs-launch pair"),
    ("find the terminal", "search_files", "search-vs-launch pair"),

    # launch_app
    ("open chrome", "launch_app", "launch-vs-search pair"),
    ("launch firefox", "launch_app", "launch-vs-search pair"),
    ("start notepad", "launch_app", "launch"),
    ("open the calculator", "launch_app", "launch-vs-search pair"),
    ("bring up the file explorer", "launch_app", "launch"),
    ("open vscode", "launch_app", "launch"),
    ("launch the terminal", "launch_app", "launch-vs-search pair"),
    ("start word", "launch_app", "launch"),
    ("fire up chrome", "launch_app", "launch"),
    ("open spotify", "launch_app", "launch"),
    ("open slack", "launch_app", "launch"),
    ("launch discord", "launch_app", "launch"),
    ("start the calculator", "launch_app", "launch-vs-search pair"),
    ("open microsoft edge", "launch_app", "launch"),

    # close_app
    ("close chrome", "close_app", "close"),
    ("quit firefox", "close_app", "close"),
    ("exit notepad", "close_app", "close"),
    ("close spotify", "close_app", "close"),
    ("get rid of firefox", "close_app", "close"),
    ("end chrome", "close_app", "close"),
    ("terminate vscode", "close_app", "close"),
    ("close the browser", "close_app", "close"),
    ("quit slack", "close_app", "close"),
    ("close discord", "close_app", "close"),
    ("exit the terminal", "close_app", "close"),

    # get_running_apps
    ("what apps are open", "get_running_apps", "list"),
    ("show running applications", "get_running_apps", "list"),
    ("list the programs running", "get_running_apps", "list"),
    ("what's currently open", "get_running_apps", "list"),
    ("which apps are running", "get_running_apps", "list"),
    ("what programs do i have open", "get_running_apps", "list"),
    ("list my running apps", "get_running_apps", "list"),
    ("what's running right now", "get_running_apps", "list"),
    ("show the active programs", "get_running_apps", "list"),
    ("which applications are open", "get_running_apps", "list"),
]

# ---------------------------------------------------------------------------
# Block 8 -- abstention / no-tool examples
# ---------------------------------------------------------------------------
# A pronoun, a complaint, or a vague reference with no resolvable target.  The
# correct behaviour is to emit no tool call.  These are NOT off-topic: they are
# desktop-shaped requests that are simply underspecified.

_ABSTENTION = [
    "close that one",
    "play that",
    "open the other one",
    "do the usual",
    "the thing we talked about",
    "make it better",
    "can you sort this out",
    "why is my laptop sluggish",
    "this is annoying",
    "it's not working",
    "something's wrong with the audio",
    "handle the situation",
    "you know what i mean",
    "just make it work",
    "the volume thing again",
    "turn it down",
    "turn it up",
    "make it louder",
    "make it softer",
    "adjust the sound",
    "do something about the volume",
    "can you deal with that",
    "take care of it",
    "the last thing we did",
    "continue from earlier",
    "pick up where we left off",
    "as we discussed",
    "like last time",
    "the other one",
    "change it",
    "switch it",
    "make it stop",
    "fix this please",
    "what's going on with it",
    "it's acting up",
    "that thing from before",
    "do the same as last time",
    "open that",
    "close this",
    "run it again",
    "do it again",
    "turn the noise off",
    "handle that for me",
    "the usual setup",
    "sort the audio thing out",
]

# ---------------------------------------------------------------------------
# Block 9 -- cross-family hard negatives
# ---------------------------------------------------------------------------
# A media noun next to a volume verb, or a kill-class verb whose object decides
# the family.  These are the queries a keyword router gets wrong, which is why
# they belong in the model's training rather than in a rule.

_B9 = [
    # kill-class: object decides the family
    ("kill the audio", "mute", "object=volume"),
    ("kill the track", "media_pause", "object=media"),
    ("kill spotify", "close_app", "object=app"),
    ("kill chrome", "close_app", "object=app"),
    ("terminate chrome", "close_app", "object=app"),
    ("kill the video", "media_pause", "object=media"),

    # on/off with a noun that could be either family
    ("sound off", "mute", "terse"),
    ("sound on", "unmute", "terse"),
    ("turn off the music", "media_pause", "media noun + particle"),
    ("turn on the music", "media_play", "media noun + particle"),
    ("turn off the sound", "mute", "volume noun + particle"),
    ("turn on the sound", "unmute", "volume noun + particle"),
    ("turn off the audio", "mute", "volume noun + particle"),
    ("turn on the audio", "unmute", "volume noun + particle"),

    # mute/unmute with a media noun (volume verb, media noun)
    ("mute the music", "mute", "volume verb, media noun"),
    ("unmute the music", "unmute", "volume verb, media noun"),
    ("mute the video", "mute", "volume verb, media noun"),

    # media verbs on an audio/sound noun (media verb, volume-ish noun)
    ("pause the sound", "media_pause", "media verb, volume noun"),
    ("resume the sound", "media_play", "media verb, volume noun"),
    ("stop the sound", "mute", "colloquial: stop == mute"),
    ("restart the video", "media_play", "media verb, media noun"),
    ("rewind the video", "media_previous", "direction, media noun"),
    ("skip the video", "media_next", "direction, media noun"),

    # bring back: restore playback vs restore audibility
    ("bring back the sound", "unmute", "restore audibility"),
    ("bring back the music", "media_play", "restore playback"),
]

# ---------------------------------------------------------------------------
# Block 10 -- volume/media boundary, within a single noun
# ---------------------------------------------------------------------------
# One noun ("audio"), two verb families.  The verb, not the noun, decides the
# family, and the same split is repeated on "music" and "speakers".

_B10 = [
    # audio + media verbs -> media family
    ("pause the audio", "media_pause", "boundary pair"),
    ("resume the audio", "media_play", "boundary pair"),
    ("stop the audio", "media_pause", "boundary pair"),
    ("play the audio", "media_play", "boundary pair"),
    ("hold the audio", "media_pause", "boundary pair"),
    ("skip the audio", "media_next", "boundary pair"),
    ("restart the audio", "media_play", "boundary pair"),
    ("rewind the audio", "media_previous", "boundary pair"),

    # audio + volume verbs -> volume family
    ("mute the audio", "mute", "boundary pair"),
    ("unmute the audio", "unmute", "boundary pair"),
    ("silence the audio", "mute", "boundary pair"),
    ("what's the audio level", "get_volume", "boundary pair"),
    ("set the audio to 30", "set_volume", "boundary pair"),
    ("lower the audio to 10", "set_volume", "boundary pair"),
    ("raise the audio to 90", "set_volume", "boundary pair"),
    ("the audio is muted", "get_volume", "boundary pair"),
    ("is the audio muted", "get_volume", "boundary pair"),

    # music + media vs volume
    ("pause the music", "media_pause", "boundary pair"),
    ("resume the music", "media_play", "boundary pair"),
    ("stop the music now", "media_pause", "boundary pair"),
    ("set the music volume to 35", "set_volume", "boundary pair"),
    ("what's the music volume", "get_volume", "boundary pair"),

    # speakers: only volume verbs make sense
    ("mute the speakers now", "mute", "boundary pair"),
    ("unmute the speakers now", "unmute", "boundary pair"),
    ("set the speakers to 40", "set_volume", "boundary pair"),
    ("the speakers are muted", "get_volume", "boundary pair"),
    ("how loud are the speakers now", "get_volume", "boundary pair"),
]


CONTRAST_BLOCKS = {
    "B1_play_vs_pause": _B1,
    "B2_mute_vs_unmute": _B2,
    "B3_next_vs_previous": _B3,
    "B4_system_intra_family": _B4,
    "B5_battery_indirect": _B5,
    "B5b_disk_cpu_ram_indirect": _B4B + _B4_INDIRECT,
    "B6_volume_read_vs_write": _B6,
    "B7_search_vs_launch": _B7,
    "B8_abstention": [(q, "__none__", "no-tool") for q in _ABSTENTION],
    "B9_cross_family_hard_negatives": _B9,
    "B10_volume_media_boundary": _B10,
}

# ---------------------------------------------------------------------------
# Off-topic negatives
# ---------------------------------------------------------------------------
# No supported desktop action and no ambiguity -- the answer is "not something
# I do".  Deliberately disjoint from ``EXTRA_NEGATIVES`` in the V4 builder and
# from the burned set.
OFF_TOPIC = [
    "explain the difference between a list and a tuple",
    "how do i center a div",
    "what's the time complexity of quicksort",
    "help me choose a laptop for gaming",
    "write a haiku about autumn",
    "what is the square root of 144",
    "how do i get better at chess",
    "summarize the history of the roman empire",
    "translate good morning into french",
    "what's a good podcast about history",
    "how do i fix a leaking tap",
    "explain the offside rule in football",
    "what should i cook for dinner",
    "how many people live in brazil",
    "what's the best way to learn guitar",
    "explain what a blockchain is",
    "how do i change a car tyre",
    "recommend a movie for tonight",
    "what's the tallest mountain in africa",
    "how do i make sourdough bread",
    "explain the water cycle to a child",
    "what's the difference between affect and effect",
    "how do i write a cover letter",
    "what causes thunder",
    "explain compound interest with an example",
    "how do i train for a marathon",
    "what's the capital of australia",
    "how do i remove a red wine stain",
    "explain how a combustion engine works",
    "what's a good book on stoicism",
    "how do i improve my typing speed",
    "explain the rules of cricket",
    "what's the population of tokyo",
    "how do i grow tomatoes in pots",
    "explain how gps works",
    "what is the best time to visit japan",
    "how do i meditate properly",
    "explain the plot of hamlet",
    "what's a synonym for happy",
    "how do i tie a bowline knot",
]
