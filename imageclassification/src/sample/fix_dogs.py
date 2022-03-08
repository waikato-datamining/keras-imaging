import os
from glob import iglob
from io import BytesIO

from xml.etree.ElementTree import ElementTree as XMLTree

from defusedxml import ElementTree

for xml_filename in iglob("annotation/Annotation/**/*", recursive=True):
    if os.path.isdir(xml_filename):
        continue
    xml: XMLTree = ElementTree.parse(xml_filename)
    path, basename = os.path.split(xml_filename)
    xml.find("folder").text = ""
    xml.find("filename").text = f"{basename}.jpg"
    for object in xml.findall("object"):
        object.find("pose").text = "Unspecified"
    root = xml.getroot()
    root.remove(root.find("source"))
    root.remove(root.find("segment"))
    new_filename = os.path.join("voc", f"{basename}.xml")
    buffer = BytesIO()
    xml.write(buffer, encoding="utf-8", short_empty_elements=False)
    buffer.seek(0)
    xml_string = buffer.read().replace(b'\n', b'').replace(b'\t', b'')
    with open(new_filename, "wb") as file:
        file.write(xml_string)
