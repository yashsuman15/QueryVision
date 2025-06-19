def crop_image(self, image, x, y, width, height):
        """Crop image based on coordinates"""
        if image is None:
            return None
        
        try:
            # Ensure coordinates are within image bounds
            img_width, img_height = image.size
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            width = max(1, min(width, img_width - x))
            height = max(1, min(height, img_height - y))
            
            # Crop the image
            cropped = image.crop((x, y, x + width, y + height))
            return cropped
            
        except Exception as e:
            print(f"Crop error: {e}")
            return None
import gradio as gr
import numpy as np
from PIL import Image
import torch
from io import BytesIO
import base64

# Import your existing modules
from models.model_loader import Owlv2ModelLoader, OwlvitModelLoader
from models.modelpredictor import ModelPredictor
from utils.visualization import annotate_image

class GradioDetectionApp:
    def __init__(self):
        self.current_model = None
        self.current_predictor = None
        self.model_type = None
        
    def load_model(self, model_choice):
        """Load the selected model"""
        try:
            if model_choice == "OWLv2":
                if self.model_type != "OWLv2":
                    self.current_model = Owlv2ModelLoader()
                    model, processor = self.current_model.get_components()
                    self.current_predictor = ModelPredictor(model, processor)
                    self.model_type = "OWLv2"
                    return "✅ OWLv2 model loaded successfully!"
                else:
                    return "✅ OWLv2 model already loaded"
            else:  # OWLvit
                if self.model_type != "OWLvit":
                    self.current_model = OwlvitModelLoader()
                    model, processor = self.current_model.get_components()
                    self.current_predictor = ModelPredictor(model, processor)
                    self.model_type = "OWLvit"
                    return "✅ OWLvit model loaded successfully!"
                else:
                    return "✅ OWLvit model already loaded"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def text_based_detection(self, image, text_queries, model_choice):
        """Perform text-based object detection"""
        if image is None:
            return None, "Please upload an image"
        
        if not text_queries.strip():
            return None, "Please enter text queries (comma-separated)"
        
        # Load model if needed
        status = self.load_model(model_choice)
        if "Error" in status:
            return None, status
        
        try:
            # Parse text queries
            queries = [q.strip() for q in text_queries.split(',') if q.strip()]
            
            # Run detection
            boxes, scores, labels = self.current_predictor.text_based_detection(image, queries)
            
            if len(boxes) == 0:
                return image, "No detections found"
            
            # Annotate image
            annotated_image = annotate_image(image, boxes, scores, labels, show_labels=True)
            
            # Create results summary
            results_text = f"Detected {len(boxes)} objects:\n"
            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                results_text += f"{i+1}. {label}: {score:.3f} confidence\n"
            
            return annotated_image, results_text
            
        except Exception as e:
            return None, f"❌ Detection error: {str(e)}"
    
    def parse_bbox_coords(self, bbox_string):
        """Parse bounding box coordinates from string"""
        try:
            coords = [int(x.strip()) for x in bbox_string.split(',')]
            if len(coords) != 4:
                return 0, 0, 100, 100
            return coords[0], coords[1], coords[2], coords[3]
        except:
            return 0, 0, 100, 100
    
    def crop_from_bbox(self, image, bbox_coords):
        """Crop image based on bounding box coordinates string"""
        if image is None:
            return None
        
        x, y, width, height = self.parse_bbox_coords(bbox_coords)
        return self.crop_image(image, x, y, width, height)
    
    def image_based_detection(self, target_image, source_image, bbox_coords, model_choice):
        """Perform image-based object detection using bounding box cropped source"""
        if target_image is None:
            return None, None, "Please upload a target image"
        
        if source_image is None:
            return None, None, "Please upload a source image"
        
        # Load model if needed
        status = self.load_model(model_choice)
        if "Error" in status:
            return None, None, status
        
        try:
            # Parse bounding box coordinates and crop
            x, y, width, height = self.parse_bbox_coords(bbox_coords)
            source_crop = self.crop_image(source_image, x, y, width, height)
            
            if source_crop is None:
                return None, None, "Failed to crop source image"
            
            # Run detection
            boxes, scores, labels = self.current_predictor.image_based_detection(target_image, source_crop)
            
            if len(boxes) == 0:
                return target_image, source_crop, "No detections found"
            
            # For image-based detection, labels are just indices
            # Convert to more meaningful labels
            text_labels = [f"Object_{i}" for i in labels]
            
            # Annotate image
            annotated_image = annotate_image(target_image, boxes, scores, text_labels, show_labels=True)
            
            # Create results summary
            results_text = f"Detected {len(boxes)} objects:\n"
            for i, (box, score) in enumerate(zip(boxes, scores)):
                results_text += f"{i+1}. Object: {score:.3f} confidence\n"
            
            return annotated_image, source_crop, results_text
            
        except Exception as e:
            return None, None, f"❌ Detection error: {str(e)}"
    
    def create_bbox_canvas(self, image):
        """Create an interactive canvas for bounding box drawing"""
        if image is None:
            return "<div>Please upload a source image first</div>"
        
        # Convert PIL image to base64 for HTML canvas
        import io
        import base64
        
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        canvas_html = f"""
        <div style="position: relative; display: inline-block;">
            <canvas id="bbox-canvas" 
                    width="{image.width}" 
                    height="{image.height}"
                    style="border: 2px solid #ddd; cursor: crosshair; max-width: 100%; height: auto;"
                    onmousedown="startDrawing(event)"
                    onmousemove="draw(event)"
                    onmouseup="stopDrawing(event)">
            </canvas>
        </div>
        
        <script>
        let canvas = document.getElementById('bbox-canvas');
        let ctx = canvas.getContext('2d');
        let img = new Image();
        let isDrawing = false;
        let startX, startY, currentX, currentY;
        
        img.onload = function() {{
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }};
        img.src = 'data:image/png;base64,{img_str}';
        
        function startDrawing(e) {{
            isDrawing = true;
            let rect = canvas.getBoundingClientRect();
            let scaleX = canvas.width / rect.width;
            let scaleY = canvas.height / rect.height;
            
            startX = (e.clientX - rect.left) * scaleX;
            startY = (e.clientY - rect.top) * scaleY;
        }}
        
        function draw(e) {{
            if (!isDrawing) return;
            
            let rect = canvas.getBoundingClientRect();
            let scaleX = canvas.width / rect.width;
            let scaleY = canvas.height / rect.height;
            
            currentX = (e.clientX - rect.left) * scaleX;
            currentY = (e.clientY - rect.top) * scaleY;
            
            // Clear and redraw
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            // Draw rectangle
            ctx.strokeStyle = '#ff0000';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
        }}
        
        function stopDrawing(e) {{
            if (!isDrawing) return;
            isDrawing = false;
            
            // Calculate final coordinates
            let x = Math.min(startX, currentX);
            let y = Math.min(startY, currentY);
            let width = Math.abs(currentX - startX);
            let height = Math.abs(currentY - startY);
            
            // Update the bbox coordinates input
            let bboxInput = document.querySelector('input[placeholder*="x,y,width,height"]');
            if (bboxInput) {{
                bboxInput.value = Math.round(x) + ',' + Math.round(y) + ',' + Math.round(width) + ',' + Math.round(height);
                bboxInput.dispatchEvent(new Event('input'));
            }}
        }}
        </script>
        """
        
        return canvas_html

# Initialize the app
app = GradioDetectionApp()

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Object Detection App", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🔍 Object Detection App")
        gr.Markdown("Choose your detection method and model to get started!")
        
        # Model selection
        with gr.Row():
            model_choice = gr.Radio(
                choices=["OWLv2", "OWLvit"], 
                value="OWLv2", 
                label="Select Model",
                info="Choose the detection model to use"
            )
        
        # Task selection tabs
        with gr.Tabs() as tabs:
            # Text-based detection tab
            with gr.TabItem("📝 Text-based Detection"):
                gr.Markdown("### Upload an image and describe what you want to detect")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        text_input_image = gr.Image(
                            type="pil",
                            label="Upload Image",
                            height=400
                        )
                        text_queries = gr.Textbox(
                            label="Text Queries",
                            placeholder="Enter objects to detect (comma-separated): car, person, dog",
                            lines=2
                        )
                        text_detect_btn = gr.Button("🔍 Detect Objects", variant="primary")
                    
                    with gr.Column(scale=1):
                        text_output_image = gr.Image(
                            label="Detection Results",
                            height=400
                        )
                        text_results = gr.Textbox(
                            label="Detection Summary",
                            lines=8,
                            max_lines=10
                        )
            
            # Image-based detection tab  
            with gr.TabItem("🖼️ Image-based Detection"):
                gr.Markdown("### Upload a target image and draw a bounding box around the object you want to find")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Target Image** (where to search)")
                        image_target = gr.Image(
                            type="pil",
                            label="Target Image",
                            height=350
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("**Source Image** (draw bounding box around object)")
                        image_source_full = gr.Image(
                            type="pil",
                            label="Source Image - Click to draw bounding box",
                            height=350
                        )
                
                # Bounding box drawing interface
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Interactive Bounding Box** - Click and drag on the source image above")
                        bbox_canvas = gr.HTML(
                            value="<div id='bbox-canvas'></div>",
                            label="Bounding Box Canvas"
                        )
                        
                        # Hidden inputs to store bbox coordinates
                        bbox_coords = gr.Textbox(
                            value="0,0,100,100",
                            label="Bounding Box (x,y,width,height)",
                            interactive=True,
                            info="Format: x,y,width,height or draw on image above"
                        )
                        
                        with gr.Row():
                            crop_btn = gr.Button("✂️ Preview Crop", variant="secondary")
                            reset_bbox_btn = gr.Button("🔄 Reset Box", variant="secondary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        cropped_preview = gr.Image(
                            label="Cropped Object Preview",
                            height=200
                        )
                    with gr.Column(scale=1):
                        gr.Markdown("""
                        **How to use:**
                        1. Upload your source image
                        2. **Method 1:** Click and drag on the image to draw a bounding box
                        3. **Method 2:** Manually enter coordinates in format: x,y,width,height
                        4. Click 'Preview Crop' to see the selected region
                        5. Adjust the box if needed
                        6. Click 'Detect Similar Objects'
                        
                        **Tips:**
                        - Draw tight boxes around objects for better results
                        - Use the preview to verify your selection
                        """)
                
                with gr.Row():
                    image_detect_btn = gr.Button("🔍 Detect Similar Objects", variant="primary")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_output = gr.Image(
                            label="Detection Results",
                            height=400
                        )
                    with gr.Column(scale=1):
                        image_results = gr.Textbox(
                            label="Detection Summary",
                            lines=8,
                            max_lines=10
                        )
        
        # Event handlers
        text_detect_btn.click(
            fn=app.text_based_detection,
            inputs=[text_input_image, text_queries, model_choice],
            outputs=[text_output_image, text_results]
        )
        
        # Update canvas when source image changes
        image_source_full.change(
            fn=app.create_bbox_canvas,
            inputs=[image_source_full],
            outputs=[bbox_canvas]
        )
        
        # Crop preview handler
        crop_btn.click(
            fn=app.crop_from_bbox,
            inputs=[image_source_full, bbox_coords],
            outputs=[cropped_preview]
        )
        
        # Reset bounding box
        reset_bbox_btn.click(
            fn=lambda: "0,0,100,100",
            outputs=[bbox_coords]
        )
        
        # Image detection handler
        image_detect_btn.click(
            fn=app.image_based_detection,
            inputs=[image_target, image_source_full, bbox_coords, model_choice],
            outputs=[image_output, cropped_preview, image_results]
        )
        
        # Add examples
        with gr.Accordion("📋 Examples", open=False):
            gr.Markdown("""
            ### Text-based Detection Examples:
            - **Queries**: `car, traffic light, person`
            - **Queries**: `dog, cat, bird`
            - **Queries**: `chair, table, laptop`
            
            ### Image-based Detection:
            1. Upload a target image (e.g., a scene with multiple objects)
            2. Upload a source image and **crop out** the specific object you want to find
            3. The model will find similar objects in the target image
            
            ### Tips:
            - For better results, use clear, high-quality images
            - Crop objects tightly in image-based detection
            - Try different models (OWLv2 vs OWLvit) for comparison
            """)
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        share=True,  # Set to False if you don't want a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860
    )