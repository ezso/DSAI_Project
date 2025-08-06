from kraken import binarization, rpred
from kraken.lib import models as kraken_models
import cv2
from PIL import Image
import xml.etree.ElementTree as ET
from kraken.pageseg import Segmentation, BBoxLine


def ocr2text(image_path, xml_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pil_image = Image.fromarray(img)
    bin_img = binarization.nlbin(pil_image)
    segmentation = get_segmentation(xml_path)
    predictions = get_predictions(bin_img, segmentation)
    line_texts = [line.prediction for line in predictions]
    full_text = "\n".join(line_texts)
    return full_text

def xml2text(xml_path):
    # ALTO XML uses a default namespace
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#'}

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []

    # Traverse each TextLine in the XML
    for text_line in root.findall('.//alto:TextLine', ns):
        words = [string.attrib.get("CONTENT", "") for string in text_line.findall("alto:String", ns)]
        line_text = " ".join(words)
        if line_text.strip():
            lines.append(line_text)

    full_text = "\n".join(lines)

    return full_text
    
def get_segmentation(xml_path):
    """
    Extracts text lines from an image using the provided XML metadata.
    Returns a Segmentation object containing the bounding boxes of text lines.
    """
     # Load and binarize the image
     # --- Parse ALTO XML ---
    ns = {'alto': 'http://www.loc.gov/standards/alto/ns-v3#'}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    segments = []
    # --- Iterate over TextLine elements ---
    for textline in root.findall('.//alto:TextLine', ns):
        hpos = int(textline.attrib['HPOS'])
        vpos = int(textline.attrib['VPOS'])
        width = int(textline.attrib['WIDTH'])
        height = int(textline.attrib['HEIGHT'])

        # Skip if width or height is zero or negative
        if width <= 0 or height <= 0:
            print(f"⚠️ Skipping textline with non-positive dimensions: width={width}, height={height}")
            continue

        x0, y0 = hpos, vpos
        x1, y1 = hpos + width, vpos + height

        # Fix swapped coordinates if necessary
        if x1 <= x0:
            x0, x1 = x1, x0
        if y1 <= y0:
            y0, y1 = y1, y0

        # Skip if box is still invalid after fixing
        if x1 <= x0 or y1 <= y0:
            print(f"⚠️ Skipping unrecoverable bounding box: {(x0, y0, x1, y1)}")
            continue

        bbox = [x0, y0, x1, y1]
        line = BBoxLine(id=None, bbox=bbox, text=None, base_dir=None, type='bbox', imagename=None, tags=None, split=None, regions=None, text_direction='horizontal-lr')
        segments.append(line)

    # Now wrap in a Segmentation object
    segmentation = Segmentation(type='bbox', imagename=None, text_direction='horizontal-lr', script_detection=False, lines=segments)
    return segmentation

def get_predictions(bin_img, segmentation):
    # Load Fraktur model directly from Zenodo
    model = kraken_models.load_any("./7933402/austriannewspapers.mlmodel")
    predictions = rpred.rpred(model, bin_img, segmentation)
    return predictions