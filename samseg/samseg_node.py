import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import torch
from sam2.build_sam import build_sam2

import sam2.build_sam
print(sam2.build_sam.__file__)  # This will print the file path of the module being used.


from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image as PILImage

import cv2

import logging
from hydra import initialize, compose
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
# from hydra.core.global_hydra import GlobalHydra

# print(GlobalHydra().is_initialized())

# if GlobalHydra().is_initialized():
#     GlobalHydra().clear()

class SAM2SegmentationNode(Node):

    def __init__(self):
        super().__init__('sam2_segmentation_node')
        self.subscription = self.create_subscription(
            Image,
            '/oak/rgb/image_raw',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(Image, '/seg/lane', 10)

        self.bridge = CvBridge()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model()
        self.predictor = SAM2ImagePredictor(self.model)

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam2_checkpoint = "../checkpoints/sam2_hiera_tiny.pt"
        model_cfg = "sam2_hiera_t.yaml"

        return build_sam2(config_file=model_cfg, ckpt_path=sam2_checkpoint, device=device)
    
    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        image = PILImage.fromarray(cv_image)
        image = np.array(image.convert("RGB"))

        self.predictor.set_image(image)
        input_point = np.array([[427, 400], [427, 420]])  # Example points
        input_label = np.array([1, 1])

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        mask_image = self.show_mask(masks[0])

        overlaid_image = self.overlay_mask_on_image(image, mask_image, alpha=1.0)

        overlaid_image_with_points = self.draw_points(overlaid_image, input_point, input_label)

        output_msg = self.bridge.cv2_to_imgmsg(overlaid_image_with_points, encoding="rgb8")
        self.publisher.publish(output_msg)


    


    def show_mask(self, mask, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
        return mask_image
    

    def overlay_mask_on_image(self, image, mask_image, alpha=1.0):
        """Overlay the mask image on the original image without transparency."""
        overlay = mask_image.astype(np.float32)
        overlaid_image = image.astype(np.float32) / 255.0
        
        # Apply the mask directly with no blending if alpha is 1.0
        if alpha == 1.0:
            overlaid_image[mask_image[:, :, 3] > 0] = overlay[mask_image[:, :, 3] > 0][:, :3]
        else:
            overlaid_image = (1 - alpha) * overlaid_image + alpha * overlay
        
        overlaid_image = (overlaid_image * 255).astype(np.uint8)
        return overlaid_image

        
    def draw_points(self, image, coords, labels):
        """Draw the points on the image."""
        for i, (x, y) in enumerate(coords):
            color = (0, 255, 0) if labels[i] == 1 else (255, 0, 0)
            cv2.circle(image, (int(x), int(y)), radius=5, color=color, thickness=-1)
            cv2.circle(image, (int(x), int(y)), radius=7, color=(255, 255, 255), thickness=1)
        return image


def build_sam2(
        config_file,
        ckpt_path=None,
        device="cuda",
        mode="eval",
        hydra_overrides_extra=[],
        apply_postprocessing=True,
        **kwargs
    ):
        print("Inside the updated build_sam2 function")
        print("Arguments received:", {
            "config_file": config_file,
            "ckpt_path": ckpt_path,
            "device": device,
            "mode": mode,
            "hydra_overrides_extra": hydra_overrides_extra,
            "apply_postprocessing": apply_postprocessing,
            "additional_kwargs": kwargs
        })
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ]
        # Read config and init model
        cfg = compose(config_name=config_file, overrides=hydra_overrides_extra)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, ckpt_path)
        model = model.to(device)
        if mode == "eval":
            model.eval()
        return model

def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")

def main(args=None):
    rclpy.init(args=args)
    sam2_segmentation_node = SAM2SegmentationNode()
    rclpy.spin(sam2_segmentation_node)
    sam2_segmentation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()