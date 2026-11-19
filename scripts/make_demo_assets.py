"""Generate simple original placeholder art for local smoke tests."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SIZE = (512, 768)
COLORS = [(226, 125, 82, 255), (80, 151, 185, 255), (157, 114, 190, 255)]


def layer() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", SIZE, (0, 0, 0, 0))
    return image, ImageDraw.Draw(image)


def save(image: Image.Image, folder: Path, name: str) -> None:
    image.save(folder / f"{name}.png")


def make_character(folder: Path, color: tuple[int, int, int, int]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    im, d = layer(); d.rounded_rectangle((135, 330, 377, 745), 75, fill=color, outline=(48, 44, 60, 255), width=8); save(im, folder, "body")
    im, d = layer(); d.ellipse((80, 70, 432, 440), fill=(248, 204, 156, 255), outline=(48, 44, 60, 255), width=8); save(im, folder, "head")
    im, d = layer(); d.ellipse((160, 210, 215, 270), fill=(255,255,255,255)); d.ellipse((295,210,350,270),fill=(255,255,255,255)); save(im, folder, "eyes_open")
    im, d = layer(); d.arc((158, 225, 218, 266), 0, 180, fill=(40,35,40,255), width=6); d.arc((292,225,352,266),0,180,fill=(40,35,40,255),width=6); save(im, folder, "eyes_closed")
    im, d = layer(); d.ellipse((180, 225, 205, 257), fill=(35,45,65,255)); d.ellipse((315,225,340,257),fill=(35,45,65,255)); save(im, folder, "pupils")
    for name, color_brow in [("eyebrows_neutral",(65,45,35,255)),("eyebrows_happy",(65,45,35,255)),("eyebrows_angry",(120,30,35,255)),("eyebrows_sad",(50,60,90,255))]:
        im,d=layer(); d.line((158,192,220,185),fill=color_brow,width=10); d.line((290,185,352,192),fill=color_brow,width=10); save(im,folder,name)
    for shape in "ABCDEFGH":
        im,d=layer(); width=26+4*(ord(shape)-ord('A')); d.ellipse((256-width//2, 335, 256+width//2, 335+18+2*(ord(shape)-ord('A'))), fill=(115,32,48,255)); save(im,folder,f"mouth_{shape}")
    im,d=layer(); d.arc((238,333,274,340),0,180,fill=(95,48,46,255),width=4); save(im,folder,"mouth_X")
    for name in ("hand_idle", "hand_point", "hand_think", "hand_wave"):
        im,d=layer();
        if name == "hand_wave": d.ellipse((390,410,475,495),fill=(248,204,156,255),outline=(48,44,60,255),width=6)
        elif name == "hand_point": d.line((360,480,470,425),fill=(248,204,156,255),width=28)
        elif name == "hand_think": d.ellipse((340,300,390,350),fill=(248,204,156,255),outline=(48,44,60,255),width=5)
        save(im,folder,name)


def make_background(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    im=Image.new("RGBA",(1920,1080),(24,35,61,255)); d=ImageDraw.Draw(im)
    for y in range(0,1080,8):
        c=30+int(30*y/1080); d.rectangle((0,y,1920,y+8),fill=(c,45+c//2,80+c,255))
    d.ellipse((520,100,1400,980),fill=(50,74,107,255),outline=(90,125,165,255),width=12)
    save(im,folder,"bg_far")
    im=Image.new("RGBA",(1920,1080),(0,0,0,0)); d=ImageDraw.Draw(im)
    d.rectangle((0,800,1920,1080),fill=(30,34,50,255)); d.line((0,800,1920,800),fill=(115,154,196,255),width=8); save(im,folder,"bg_mid")
    im=Image.new("RGBA",(1920,1080),(0,0,0,0)); d=ImageDraw.Draw(im)
    for x in (100,1820): d.rounded_rectangle((x-25,120,x+25,600),20,fill=(245,199,112,90))
    save(im,folder,"lighting")


def main() -> None:
    for i, color in enumerate(COLORS, 1): make_character(ROOT / "assets" / "characters" / f"char_{i}", color)
    make_background(ROOT / "assets" / "backgrounds")
    print("Generated original layered demo assets.")

if __name__ == "__main__": main()
