# AutoYOLO - Ultralytics YOLOv8 Web UI v2

[![Gradient](https://assets.paperspace.io/img/gradient-badge.svg)](https://console.paperspace.com/github/gradient-ai/autoyolo?machine=Free-GPU)

This Gradio application is designed to facilitate the end-to-end creation of a YOLOv8 object detection model using Segment Anything, GroundingDINO, BLIP-2, and Dolly v2 to facilitate the automatic labeling of objects in images. Users can then train Ultralytics YOLOv8 models to generate predictions on inputted videos and images. Optionally, users may also manually label images as desired.

## Capabilities

- **AutoLabel**: The key contribution of this application is the AutoLabeler. Using the autolabel tab, users can automatically generate fully labeled images in the Ultralytics YOLOv8 format with only the submission of the images and desired, target, object labels
- **Manually Label Images**: this tab lets you upload images, either in bulk or one at a time, to be labeled. The bounding boxes are automatically detected, and the labels are assigned through a textbox. Entries are separated by semi-colons
- **Image Gallery**: this tab allows us to view our labeled images, seperated by the assigned training split
- **Train**: train any of the YOLOv8 models on the labeled images. Outputs the validation metrics and the best trained model from the run, `best.pt`
- **Inference**: predict object labels on images and videos. Works for direct upload and URL submission of images and YouTube Videos

## Next steps

- Integrating with RoboFlow to enable training on the application with existing projects and Universe datasets
- Streaming video object detection for real time viewing and interaction with the object detection model
- Add in additional text models (GPT4All, OpenAssistant, Otter, etc.) to enable multimodal integration. Potentially removes BLIP-2 from pipeline and speeds up processing

## Thanks and credits to:

- This application was inspired by the work done by Idea Research on their [Grounded Segment Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) project. Thanks to them for releasing their awesome work, and for inspiring this project.
- This application wouldn't have been feasible without the groundwork completed by the researchers for the [GLIGEN](https://github.com/gligen/GLIGEN) project. Their bounding box detector code was instrumental to making this work.
- [Ultralytics](https://github.com/ultralytics/ultralytics) for their incredible work on YOLOv8
