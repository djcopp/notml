"""A CHIP-8 computer compiled into an ONNX graph.

``machine.build(cycles, batch)`` returns a model that runs ``cycles`` instructions on every console in a
batch and then ticks the 60 Hz timers once, so calling it once per frame runs the machine at
``60 * cycles`` instructions a second. ``reference.Chip8`` is a plain Python emulator with the same
behaviour, used to test the graph. ``asm.assemble`` turns assembly into a ROM.

Behaviour follows the common modern conventions:
  * 8XY6 / 8XYE shift VX in place (VY is ignored)
  * FX55 / FX65 leave I unchanged
  * 8XY1 / 8XY2 / 8XY3 leave VF unchanged
  * sprites wrap their starting position but clip at the screen edge
  * FX0A takes the lowest-numbered key held down; with none held it repeats
  * CXNN uses a Lehmer generator per console: s = 75 s mod 65537, random byte = s mod 256
"""

from notml.chip8 import asm, machine, reference

__all__ = ["asm", "machine", "reference"]


def program(name: str) -> bytes:
    """Assemble one of the bundled programs (``notml/chip8/programs/<name>.asm``)."""
    import importlib.resources

    return asm.assemble(importlib.resources.files(__package__).joinpath(f"programs/{name}.asm").read_text())
