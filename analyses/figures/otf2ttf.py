# Convert CFF-based OTF to TrueType (quadratic) so matplotlib embeds as Type 42. Based on fontTools snippets/otf2ttf.py.
import sys
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
def glyphs_to_quadratic(glyphs, max_err=1.0, reverse_direction=True):
    out = {}
    for name in glyphs.keys():
        g = glyphs[name]; pen = TTGlyphPen(glyphs)
        g.draw(Cu2QuPen(pen, max_err, reverse_direction=reverse_direction)); out[name] = pen.glyph()
    return out
def otf_to_ttf(src, dst):
    f = TTFont(src); assert f.sfntVersion == "OTTO"
    glyf = newTable("glyf"); glyf.glyphOrder = f.getGlyphOrder()
    glyf.glyphs = glyphs_to_quadratic(f.getGlyphSet())
    del f["CFF "]
    if "VORG" in f: del f["VORG"]
    f["glyf"] = glyf; f["loca"] = newTable("loca")
    f["maxp"] = newTable("maxp"); f["maxp"].tableVersion = 0x00010000
    for k in ("maxZones","maxTwilightPoints","maxStorage","maxFunctionDefs","maxInstructionDefs","maxStackElements","maxSizeOfInstructions","maxComponentElements"):
        setattr(f["maxp"], k, 0 if k!="maxZones" else 1)
    f["maxp"].maxComponentDepth = 0
    f["post"].formatType = 2.0; f["post"].extraNames = []; f["post"].mapping = {}; f["post"].glyphOrder = None
    f["head"].glyphDataFormat = 0
    f.sfntVersion = "\x00\x01\x00\x00"; f.save(dst)
for s, d in zip(sys.argv[1::2], sys.argv[2::2]): otf_to_ttf(s, d)
