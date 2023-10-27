import io
import os
from dataclasses import dataclass

import numpy as np
import supervision as sv
from autodistill.detection import CaptionOntology, DetectionBaseModel
from google.cloud import vision
from PIL import Image

HOME = os.path.expanduser("~")


@dataclass
class GCPVision(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.client = vision.ImageAnnotatorClient()
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        with Image.open(input) as image:
            buffered = io.BytesIO()
            image.save(buffered, format=image.format)
            image_bytes = buffered.getvalue()

        source_image_height, source_image_width = image.size

        image = vision.Image(content=image_bytes)

        objects = self.client.object_localization(
            image=image
        ).localized_object_annotations

        xyxys, confidences, class_ids = [], [], []

        classes = self.ontology.classes()

        for object_ in objects:
            if object_.score < confidence:
                continue

            if object_.name not in classes:
                continue

            # bounding boxes must be in the format [x0, y0, x1, y1]
            # not the polygons returned by the GCP Vision API

            object_bboxes = []

            for vertex in object_.bounding_poly.normalized_vertices:
                object_bboxes.append([vertex.x, vertex.y])

            object_bboxes = np.array(object_bboxes)

            x0 = object_bboxes[:, 0].min()
            y0 = object_bboxes[:, 1].min()
            x1 = object_bboxes[:, 0].max()
            y1 = object_bboxes[:, 1].max()

            # normalize as image size, not 0-1
            x0 *= source_image_width
            y0 *= source_image_height
            x1 *= source_image_width
            y1 *= source_image_height

            xyxys.append([x0, y0, x1, y1])

            confidences.append(object_.score)

            class_ids.append(classes.index(object_.name))

        if len(xyxys) == 0:
            return sv.Detections.empty()

        return sv.Detections(
            xyxy=np.array(xyxys),
            class_id=np.array(class_ids),
            confidence=np.array(confidences),
        )
