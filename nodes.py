# ComfyUI custom nodes: S3 file uploads from file path
# Save as: custom_nodes/s3_upload/nodes.py

import io
import os
import time
import json
import mimetypes
import shutil
from typing import Optional

import requests
import torch
import numpy as np
from PIL import Image

# tkinter is optional (only needed for FileExplorer)
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# boto3 is optional (only needed for auth_mode="boto3")
try:
    import boto3
    from boto3.s3.transfer import TransferConfig
    BOTO3_AVAILABLE = True
except Exception:
    BOTO3_AVAILABLE = False


# ---------- Helpers ----------

def _clean_part(s: str) -> str:
    return str(s).strip().strip("/")

def build_s3_key(user_id: str, project_type: str, project_id: str, output: str) -> str:
    user_id = _clean_part(user_id)
    project_type = _clean_part(project_type)
    project_id = _clean_part(project_id)
    output = str(output).lstrip("/")  # allow nested like "images/image1.png"
    if not (user_id and project_type and project_id and output):
        raise ValueError("user_id, project_type, project_id, and output must be non-empty")
    return f"users/{user_id}/projects/{project_type}/{project_id}/{output}"

def guess_content_type(filename: str, fallback: str) -> str:
    ctype, _ = mimetypes.guess_type(filename)
    return ctype or fallback

def _https_url(bucket: str, region: Optional[str], key: str) -> str:
    if region and region != "us-east-1":
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    return f"https://{bucket}.s3.amazonaws.com/{key}"

def _put_presigned(url: str, data: bytes, content_type: str = "application/octet-stream",
                   method: str = "PUT", fields: Optional[dict] = None) -> None:
    if method.upper() == "PUT":
        r = requests.put(url, data=data, headers={"Content-Type": content_type})
        r.raise_for_status()
    elif method.upper() == "POST":
        if not fields:
            raise ValueError("Presigned POST requires 'fields'.")
        files = {"file": (fields.get("key", "upload.bin"), data, content_type)}
        r = requests.post(url, data=fields, files=files)
        r.raise_for_status()
    else:
        raise ValueError(f"Unsupported presign method: {method}")

def _boto3_upload_bytes(bucket: str, key: str, data: bytes, region: Optional[str], content_type: str) -> None:
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 not installed. Install it or use presigned_url auth_mode.")
    session = boto3.session.Session(region_name=region) if region else boto3.session.Session()
    s3 = session.client("s3")
    cfg = TransferConfig(multipart_threshold=8*1024*1024, max_concurrency=8,
                         multipart_chunksize=8*1024*1024, use_threads=True)
    s3.upload_fileobj(
        io.BytesIO(data), bucket, key,
        ExtraArgs={"ContentType": content_type, "ChecksumAlgorithm": "SHA256"},
        Config=cfg
    )

def _boto3_upload_path(bucket: str, key: str, path: str, region: Optional[str], content_type: Optional[str]) -> None:
    if not BOTO3_AVAILABLE:
        raise RuntimeError("boto3 not installed. Install it or use presigned_url auth_mode.")
    session = boto3.session.Session(region_name=region) if region else boto3.session.Session()
    s3 = session.client("s3")
    cfg = TransferConfig(multipart_threshold=8*1024*1024, max_concurrency=8,
                         multipart_chunksize=8*1024*1024, use_threads=True)
    extra = {"ChecksumAlgorithm": "SHA256"}
    if content_type:
        extra["ContentType"] = content_type
    s3.upload_file(path, bucket, key, ExtraArgs=extra, Config=cfg)


# ---------- ComfyUI Nodes ----------

class S3UploadMedia:
    """
    Simple S3 upload node that uploads any file from a file path.
    Auto-detects file type from extension and uploads to S3.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": ""}),
                "s3_key": ("STRING", {"default": ""}),
                "bucket": ("STRING", {"default": "clipizy"}),
                "region": ("STRING", {"default": "eu-north-1"}),
                "auth_mode": (["presigned_url", "boto3"],),
            },
            "optional": {
                "presign_endpoint": ("STRING", {"default": "https://api.yourapp.com/storage/presign"}),
                "presign_method": (["PUT", "POST"], {"default": "PUT"}),
                "presign_headers_json": ("STRING", {"default": ""}),
                "content_type": ("STRING", {"default": ""}),  # auto-guessed if empty
                "force_type": (["auto", "image", "video", "audio", "file"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("s3_key", "https_url")
    FUNCTION = "upload_media"
    CATEGORY = "S3/Upload"
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        # Accept all inputs - validation happens in the function
        return True
    
    def upload_media(self, file_path, s3_key, bucket, region, auth_mode, presign_endpoint=None, presign_method="PUT",
                    presign_headers_json="", content_type="", force_type="auto"):
        
        ts = int(time.time())
        detected_type = "file"
        file_extension = "bin"

        # --- Validate file path ---
        if not file_path or not file_path.strip():
            raise ValueError("file_path is required and cannot be empty.")
        
        file_path = file_path.strip()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # --- Auto-detect file type from extension ---
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.3gp']:
            detected_type, file_extension = "video", file_ext[1:]
        elif file_ext in ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus']:
            detected_type, file_extension = "audio", file_ext[1:]
        elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp', '.svg']:
            detected_type, file_extension = "image", file_ext[1:]
        else:
            detected_type, file_extension = "file", file_ext[1:] or "bin"

        # --- Override if force_type is explicitly set ---
        if force_type != "auto":
            detected_type = force_type
            file_extension = {"image": "png", "video": "mp4", "audio": "mp3", "file": "bin"}.get(force_type, file_extension)

        # --- Build S3 key ---
        # Use the provided s3_key, or fallback to filename if empty
        if not s3_key or not s3_key.strip():
            # Fallback: use filename from file_path
            s3_key = os.path.basename(file_path)
        else:
            # Use provided s3_key, clean it up
            s3_key = s3_key.strip()
            # Remove leading slash if present
            if s3_key.startswith('/'):
                s3_key = s3_key[1:]
        
        # Get filename for content type detection
        filename = os.path.basename(file_path)
        if not content_type:
            content_type = guess_content_type(filename, "application/octet-stream")

        # --- Upload ---
        if auth_mode == "presigned_url":
            if not presign_endpoint:
                raise ValueError("presign_endpoint is required for presigned_url auth_mode")
            headers = {}
            if presign_headers_json.strip():
                headers = json.loads(presign_headers_json)
            payload = {
                "bucket": bucket,
                "region": region,
                "key": s3_key,
                "content_type": content_type,
                "method": presign_method
            }
            r = requests.post(presign_endpoint, json=payload, headers=headers, timeout=60)
            r.raise_for_status()
            presign = r.json()
            url = presign.get("url")
            method = (presign.get("method") or presign_method or "PUT").upper()
            fields = presign.get("fields")
            if not url:
                raise RuntimeError("Presign response missing 'url'")
            
            # Read file data for presigned upload
            with open(file_path, "rb") as f:
                data = f.read()
            _put_presigned(url, data, content_type, method=method, fields=fields)

        elif auth_mode == "boto3":
            _boto3_upload_path(bucket, s3_key, file_path, region, content_type)
        else:
            raise ValueError("Unknown auth_mode")

        return (s3_key, _https_url(bucket, region, s3_key))


class S3FileExplorer:
    """
    File explorer node that opens a file dialog to select any file type.
    Returns the selected file path as a string.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_types": (["All Files", "Images", "Videos", "Audio", "Documents", "Custom"], {"default": "All Files"}),
                "custom_extensions": ("STRING", {"default": "*.txt;*.csv;*.json"}),
                "initial_directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "title": ("STRING", {"default": "Select File"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "select_file"
    CATEGORY = "S3/File Selection"
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    def select_file(self, file_types, custom_extensions, initial_directory, title="Select File"):
        """
        Open file dialog and return selected file path.
        """
        if not TKINTER_AVAILABLE:
            return ("Error: tkinter not available. Please install tkinter or use file_path input directly.",)
        
        try:
            # Hide the main tkinter window
            root = tk.Tk()
            root.withdraw()
            
            # Set initial directory
            if initial_directory and os.path.exists(initial_directory):
                os.chdir(initial_directory)
            
            # Define file type filters
            filetype_filters = {
                "All Files": [("All Files", "*.*")],
                "Images": [
                    ("Image Files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff *.webp *.svg"),
                    ("PNG Files", "*.png"),
                    ("JPEG Files", "*.jpg *.jpeg"),
                    ("All Files", "*.*")
                ],
                "Videos": [
                    ("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.m4v *.3gp"),
                    ("MP4 Files", "*.mp4"),
                    ("AVI Files", "*.avi"),
                    ("All Files", "*.*")
                ],
                "Audio": [
                    ("Audio Files", "*.mp3 *.wav *.flac *.aac *.ogg *.m4a *.wma *.opus"),
                    ("MP3 Files", "*.mp3"),
                    ("WAV Files", "*.wav"),
                    ("All Files", "*.*")
                ],
                "Documents": [
                    ("Document Files", "*.pdf *.doc *.docx *.txt *.rtf"),
                    ("PDF Files", "*.pdf"),
                    ("Text Files", "*.txt"),
                    ("All Files", "*.*")
                ],
                "Custom": [
                    ("Custom Files", custom_extensions),
                    ("All Files", "*.*")
                ]
            }
            
            # Get the appropriate file types
            filetypes = filetype_filters.get(file_types, filetype_filters["All Files"])
            
            # Open file dialog
            file_path = filedialog.askopenfilename(
                title=title,
                filetypes=filetypes
            )
            
            # Clean up
            root.destroy()
            
            if file_path:
                return (file_path,)
            else:
                # Return empty string if no file selected
                return ("",)
                
        except Exception as e:
            # Return error message if something goes wrong
            return (f"Error: {str(e)}",)


class S3SaveImageExportPath:
    """
    Save IMAGE tensor to specified path and return the relative path.
    Accepts ComfyUI IMAGE input and saves as PNG file.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "input_path": ("STRING", {"default": "output/image.png"}),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("relative_path",)
    FUNCTION = "save_image"
    CATEGORY = "S3/File Export"
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    def save_image(self, image, input_path, overwrite=False):
        """
        Save IMAGE tensor to specified path and return relative path.
        """
        try:
            print(f"DEBUG: S3SaveImageExportPath called with image type: {type(image)}")
            if image is None:
                raise ValueError("image input is required.")
            
            print(f"DEBUG: Image input received: {image}")
            print(f"DEBUG: Image is None: {image is None}")
            
            # Use input_path directly
            dest_path = input_path.strip()
            if not dest_path:
                dest_path = "output/image.png"
            
            # Make path relative to ComfyUI root if not absolute
            if not os.path.isabs(dest_path):
                dest_path = os.path.join(os.getcwd(), dest_path)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(dest_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Ensure PNG extension
            if not dest_path.lower().endswith('.png'):
                dest_path += '.png'
            
            # Handle overwrite - if file exists and overwrite is False, just use the original path
            # (removed automatic numbering to keep original filename)
            
            # Convert tensor to PIL Image and save
            if isinstance(image, torch.Tensor):
                print(f"DEBUG: Image tensor shape: {image.shape}, dtype: {image.dtype}")
                print(f"DEBUG: Image tensor min/max: {image.min().item():.4f}/{image.max().item():.4f}")
                
                # Handle different tensor shapes and data types
                if image.dtype == torch.uint8:
                    # Image is already in uint8 format, just convert to numpy
                    t = image.cpu().numpy()
                    print(f"DEBUG: Image already uint8, converted to numpy - shape: {t.shape}, min/max: {t.min()}/{t.max()}")
                else:
                    # Image is in float format, clamp and convert to uint8
                    # First clamp to 0-1 range, then multiply by 255 to get 0-255 range, then convert to uint8
                    t = (image.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                    print(f"DEBUG: After clamp, multiply by 255, and uint8 conversion - shape: {t.shape}, min/max: {t.min()}/{t.max()}")
                
                if t.ndim == 4:
                    # Batch dimension present, take first image
                    if t.shape[0] == 1:
                        t = t[0]
                    else:
                        # Multiple images in batch, take first one
                        t = t[0]
                    print(f"DEBUG: After removing batch dimension - shape: {t.shape}")
                
                # Handle 3D tensors (H, W, C)
                if t.ndim == 3:
                    if t.shape[2] in (1, 3, 4):
                        mode = "RGBA" if t.shape[2] == 4 else "RGB"
                        print(f"DEBUG: Using mode: {mode} for shape: {t.shape}")
                    elif t.shape[0] in (1, 3, 4):
                        # Channel first format (C, H, W), transpose to (H, W, C)
                        t = t.transpose(1, 2, 0)
                        mode = "RGBA" if t.shape[2] == 4 else "RGB"
                        print(f"DEBUG: Transposed to channel last - shape: {t.shape}, mode: {mode}")
                    else:
                        raise ValueError(f"Unexpected tensor shape: {t.shape}")
                else:
                    raise ValueError(f"Unexpected tensor shape: {t.shape}")

                print(f"DEBUG: Final tensor shape: {t.shape}, min/max: {t.min()}/{t.max()}")
                img = Image.fromarray(t, mode=mode)
                # Save to workspace directory
                workspace_path = os.path.join("workspace", dest_path)
                img.save(workspace_path, format="PNG")
                print(f"DEBUG: Successfully saved image as PNG: {workspace_path}")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Return relative path from ComfyUI root (without workspace prefix)
            relative_path = os.path.relpath(dest_path, os.getcwd())
            
            return (relative_path,)
            
        except Exception as e:
            # Return empty string instead of error message to avoid issues with downstream nodes
            print(f"ERROR in S3SaveImageExportPath: {str(e)}")
            return ("",)


class S3SaveVideoExportPath:
    """
    Save VIDEO data to specified path and return the relative path.
    Accepts ComfyUI VIDEO input and saves as MP4 file.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video": ("VIDEO",),
                "input_path": ("STRING", {"default": "output/video.mp4"}),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("relative_path",)
    FUNCTION = "save_video"
    CATEGORY = "S3/File Export"
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    def save_video(self, video, input_path, overwrite=False):
        """
        Save VIDEO data to specified path and return relative path.
        """
        try:
            if video is None:
                raise ValueError("video input is required.")
            
            # Use input_path directly
            dest_path = input_path.strip()
            if not dest_path:
                dest_path = "output/video.mp4"
            
            # Make path relative to ComfyUI root if not absolute
            if not os.path.isabs(dest_path):
                dest_path = os.path.join(os.getcwd(), dest_path)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(dest_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Ensure MP4 extension
            if not dest_path.lower().endswith('.mp4'):
                dest_path += '.mp4'
            
            # Handle overwrite - if file exists and overwrite is False, just use the original path
            # (removed automatic numbering to keep original filename)
            
            # Handle ComfyUI video data structure
            if isinstance(video, dict):
                # Debug: Print video data structure to understand the format
                print(f"DEBUG: Video data structure keys: {list(video.keys())}")
                print(f"DEBUG: Video data structure: {video}")
                
                # Try to extract file path from video data structure first
                video_file_path = None
                
                # Check various possible keys for video file path
                for key in ["video_file", "file", "path", "filename", "video_path", "name", "src"]:
                    if key in video and video[key]:
                        video_file_path = video[key]
                        print(f"DEBUG: Found video file path in key '{key}': {video_file_path}")
                        break
                
                # If no file path found, try to get the first string value that looks like a path
                if not video_file_path:
                    for key, value in video.items():
                        if isinstance(value, str) and ("/" in value or "\\" in value or value.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))):
                            video_file_path = value
                            print(f"DEBUG: Found video file path in key '{key}': {video_file_path}")
                            break
                
                if video_file_path and os.path.exists(video_file_path):
                    # Copy the existing video file
                    print(f"DEBUG: Copying video file from: {video_file_path}")
                    workspace_path = os.path.join("workspace", dest_path)
                    shutil.copy2(video_file_path, workspace_path)
                else:
                    # Handle video data directly (like audio function does)
                    print(f"DEBUG: No file path found, trying to handle video data directly")
                    
                    # Check for common video data keys (similar to audio waveform/sample_rate)
                    video_data = None
                    video_format = "mp4"  # default format
                    
                    # Look for video tensor data
                    for key in ["video", "data", "frames", "tensor", "array"]:
                        if key in video and video[key] is not None:
                            video_data = video[key]
                            print(f"DEBUG: Found video data in key '{key}': {type(video_data)}")
                            break
                    
                    if video_data is not None:
                        # Handle tensor data (similar to audio waveform processing)
                        if isinstance(video_data, torch.Tensor):
                            print(f"DEBUG: Processing video tensor, shape: {video_data.shape}")
                            
                            # Convert tensor to numpy
                            video_np = video_data.cpu().numpy()
                            
                            # Handle different tensor shapes
                            if video_np.ndim == 5:  # [batch, time, height, width, channels]
                                if video_np.shape[0] == 1:
                                    video_np = video_np[0]  # Remove batch dimension
                                else:
                                    video_np = video_np[0]  # Take first batch
                            
                            # Ensure video is in the right range (0-255 for uint8)
                            if video_np.dtype != np.uint8:
                                if video_np.max() <= 1.0:
                                    video_np = (video_np * 255).astype(np.uint8)
                                else:
                                    video_np = video_np.astype(np.uint8)
                            
                            # Save as MP4 using OpenCV (if available) or as raw data
                            try:
                                import cv2
                                # Get video properties
                                height, width = video_np.shape[1], video_np.shape[2]
                                fps = 30  # default fps, could be extracted from video data
                                
                                # Create video writer
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                workspace_path = os.path.join("workspace", dest_path)
                                out = cv2.VideoWriter(workspace_path, fourcc, fps, (width, height))
                                
                                # Write each frame
                                for frame in video_np:
                                    # Convert from RGB to BGR for OpenCV
                                    if frame.shape[2] == 3:
                                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                    else:
                                        frame_bgr = frame
                                    out.write(frame_bgr)
                                
                                out.release()
                                print(f"DEBUG: Successfully saved video tensor as MP4 file: {dest_path}")
                                
                            except ImportError:
                                # Fallback: save as raw binary data
                                print(f"DEBUG: OpenCV not available, saving as raw binary data")
                                workspace_path = os.path.join("workspace", dest_path)
                                with open(workspace_path, "wb") as f:
                                    f.write(video_np.tobytes())
                                print(f"DEBUG: Successfully saved video tensor as raw binary: {dest_path}")
                                
                        elif hasattr(video_data, 'tobytes'):
                            # Handle numpy array or similar
                            print(f"DEBUG: Processing video array data")
                            workspace_path = os.path.join("workspace", dest_path)
                            with open(workspace_path, "wb") as f:
                                f.write(video_data.tobytes())
                            print(f"DEBUG: Successfully saved video array as MP4 file: {dest_path}")
                        else:
                            # Try to convert to bytes
                            print(f"DEBUG: Converting video data to bytes")
                            data = bytes(video_data)
                            workspace_path = os.path.join("workspace", dest_path)
                            with open(workspace_path, "wb") as f:
                                f.write(data)
                            print(f"DEBUG: Successfully saved video data as MP4 file: {dest_path}")
                    else:
                        raise ValueError("Could not find video data in video input structure.")
            else:
                # Handle VideoFromFile type directly
                print(f"DEBUG: Video type: {type(video)}")
                print(f"DEBUG: VideoFromFile attributes: {dir(video)}")
                
                # Try to get file path from VideoFromFile object
                video_file_path = None
                
                # Check all possible attributes that might contain the file path
                possible_attrs = ['filename', 'name', 'path', 'file_path', 'file', 'src', 'url', 'location']
                
                for attr in possible_attrs:
                    if hasattr(video, attr):
                        try:
                            value = getattr(video, attr)
                            print(f"DEBUG: VideoFromFile.{attr} = {value} (type: {type(value)})")
                            if value and isinstance(value, str) and (os.path.exists(value) or "/" in value or "\\" in value):
                                video_file_path = value
                                print(f"DEBUG: Found video file path in attribute '{attr}': {video_file_path}")
                                break
                        except Exception as e:
                            print(f"DEBUG: Error accessing VideoFromFile.{attr}: {e}")
                
                # If no direct file path found, try to convert the object to string
                if not video_file_path:
                    try:
                        str_value = str(video)
                        print(f"DEBUG: VideoFromFile string representation: {str_value}")
                        if str_value and ("/" in str_value or "\\" in str_value) and os.path.exists(str_value):
                            video_file_path = str_value
                            print(f"DEBUG: Found video file path from string representation: {video_file_path}")
                    except Exception as e:
                        print(f"DEBUG: Error getting string representation: {e}")
                
                if video_file_path and os.path.exists(video_file_path):
                    print(f"DEBUG: Copying video file from VideoFromFile: {video_file_path}")
                    workspace_path = os.path.join("workspace", dest_path)
                    shutil.copy2(video_file_path, workspace_path)
                else:
                    # Try to use the save_to method (this is the correct way!)
                    if hasattr(video, 'save_to'):
                        try:
                            print(f"DEBUG: Using VideoFromFile.save_to() method to save video")
                            workspace_path = os.path.join("workspace", dest_path)
                            video.save_to(workspace_path)
                            print(f"DEBUG: Successfully saved video using save_to method: {workspace_path}")
                        except Exception as e:
                            print(f"DEBUG: save_to method failed: {e}")
                            # Try to access the underlying file object if save_to fails
                            if hasattr(video, '_VideoFromFile__file'):
                                try:
                                    file_obj = video._VideoFromFile__file
                                    if file_obj and hasattr(file_obj, 'name'):
                                        file_path = file_obj.name
                                        print(f"DEBUG: Found file path from private file object: {file_path}")
                                        if os.path.exists(file_path):
                                            workspace_path = os.path.join("workspace", dest_path)
                                            shutil.copy2(file_path, workspace_path)
                                        else:
                                            raise ValueError(f"File object path does not exist: {file_path}")
                                    else:
                                        raise ValueError(f"VideoFromFile private file object has no name attribute")
                                except Exception as e2:
                                    raise ValueError(f"Could not extract file path from VideoFromFile: {e2}")
                            else:
                                raise ValueError(f"VideoFromFile has no save_to method or private file object")
                    else:
                        raise ValueError(f"VideoFromFile object has no save_to method. Available attributes: {[attr for attr in dir(video) if not attr.startswith('_')]}")
            
            # Return relative path from ComfyUI root
            relative_path = os.path.relpath(dest_path, os.getcwd())
            
            return (relative_path,)
            
        except Exception as e:
            # Return empty string instead of error message to avoid issues with downstream nodes
            print(f"ERROR in S3SaveVideoExportPath: {str(e)}")
            return ("",)


class S3SaveAudioExportPath:
    """
    Save AUDIO data to specified path and return the relative path.
    Accepts ComfyUI AUDIO input and saves as WAV file.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "input_path": ("STRING", {"default": "output/audio.wav"}),
            },
            "optional": {
                "overwrite": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("relative_path",)
    FUNCTION = "save_audio"
    CATEGORY = "S3/File Export"
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    def save_audio(self, audio, input_path, overwrite=False):
        """
        Save AUDIO data to specified path and return relative path.
        """
        try:
            if audio is None:
                raise ValueError("audio input is required.")
            
            # Use input_path directly
            dest_path = input_path.strip()
            if not dest_path:
                dest_path = "output/audio.wav"
            
            # Make path relative to ComfyUI root if not absolute
            if not os.path.isabs(dest_path):
                dest_path = os.path.join(os.getcwd(), dest_path)
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(dest_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Ensure WAV extension
            if not dest_path.lower().endswith('.wav'):
                dest_path += '.wav'
            
            # Handle overwrite - if file exists and overwrite is False, just use the original path
            # (removed automatic numbering to keep original filename)
            
            # Handle ComfyUI audio data structure
            if isinstance(audio, dict):
                # Debug: Print audio data structure to understand the format
                print(f"DEBUG: Audio data structure keys: {list(audio.keys())}")
                print(f"DEBUG: Audio data structure: {audio}")
                
                # Try to extract file path from audio data structure first
                audio_file_path = None
                
                # Check various possible keys for audio file path
                for key in ["audio_file", "file", "path", "filename", "audio_path", "name", "src"]:
                    if key in audio and audio[key]:
                        audio_file_path = audio[key]
                        print(f"DEBUG: Found audio file path in key '{key}': {audio_file_path}")
                        break
                
                # If no file path found, try to get the first string value that looks like a path
                if not audio_file_path:
                    for key, value in audio.items():
                        if isinstance(value, str) and ("/" in value or "\\" in value or value.endswith(('.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a'))):
                            audio_file_path = value
                            print(f"DEBUG: Found audio file path in key '{key}': {audio_file_path}")
                            break
                
                if audio_file_path and os.path.exists(audio_file_path):
                    # Copy the existing audio file
                    print(f"DEBUG: Copying audio file from: {audio_file_path}")
                    workspace_path = os.path.join("workspace", dest_path)
                    shutil.copy2(audio_file_path, workspace_path)
                else:
                    # Handle waveform tensor data directly
                    print(f"DEBUG: No file path found, trying to handle waveform tensor data")
                    if "waveform" in audio:
                        waveform = audio["waveform"]
                        sample_rate = audio.get("sample_rate", 44100)
                        print(f"DEBUG: Found waveform tensor, shape: {waveform.shape}, sample_rate: {sample_rate}")
                        
                        # Convert tensor to numpy and save as WAV
                        if isinstance(waveform, torch.Tensor):
                            # Convert to numpy and ensure it's in the right format
                            audio_np = waveform.cpu().numpy()
                            
                            # Handle different tensor shapes
                            if audio_np.ndim == 3:
                                # Shape: [batch, channels, samples] -> [channels, samples]
                                if audio_np.shape[0] == 1:
                                    audio_np = audio_np[0]  # Remove batch dimension
                                else:
                                    audio_np = audio_np[0]  # Take first batch
                            
                            # Ensure audio is in the right range for WAV format (-1.0 to 1.0)
                            if audio_np.dtype != np.float32:
                                audio_np = audio_np.astype(np.float32)
                            
                            # Clamp values to valid range
                            audio_np = np.clip(audio_np, -1.0, 1.0)
                            
                            # Save as WAV file using scipy or wave module
                            try:
                                import scipy.io.wavfile as wavfile
                                # Transpose to [samples, channels] format for scipy
                                if audio_np.ndim == 2:
                                    audio_np = audio_np.T
                                workspace_path = os.path.join("workspace", dest_path)
                                wavfile.write(workspace_path, sample_rate, audio_np)
                                print(f"DEBUG: Successfully saved waveform as WAV file: {dest_path}")
                            except ImportError:
                                # Fallback to wave module if scipy not available
                                import wave
                                
                                # Convert to 16-bit PCM
                                audio_16bit = (audio_np * 32767).astype(np.int16)
                                
                                # Ensure mono or stereo format
                                if audio_16bit.ndim == 1:
                                    audio_16bit = audio_16bit.reshape(-1, 1)
                                elif audio_16bit.ndim == 2 and audio_16bit.shape[0] == 1:
                                    audio_16bit = audio_16bit.T
                                
                                workspace_path = os.path.join("workspace", dest_path)
                                with wave.open(workspace_path, 'wb') as wav_file:
                                    wav_file.setnchannels(audio_16bit.shape[1])
                                    wav_file.setsampwidth(2)  # 16-bit = 2 bytes
                                    wav_file.setframerate(sample_rate)
                                    wav_file.writeframes(audio_16bit.tobytes())
                                print(f"DEBUG: Successfully saved waveform as WAV file using wave module: {dest_path}")
                        else:
                            raise ValueError(f"Waveform is not a tensor: {type(waveform)}")
                    else:
                        raise ValueError("Could not find waveform data in audio input.")
            else:
                raise ValueError(f"Unsupported audio type: {type(audio)}")
            
            # Return relative path from ComfyUI root
            relative_path = os.path.relpath(dest_path, os.getcwd())
            
            return (relative_path,)
            
        except Exception as e:
            # Return empty string instead of error message to avoid issues with downstream nodes
            print(f"ERROR in S3SaveAudioExportPath: {str(e)}")
            return ("",)


class S3CombineImagesToVideo:
    """
    Combine multiple images into a video.
    Accepts multiple IMAGE inputs and creates an MP4 video.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 30, "min": 1, "max": 120}),
                "duration_per_frame": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 10.0}),
            },
            "optional": {
                "video_codec": (["mp4v", "h264", "xvid"], {"default": "mp4v"}),
                "quality": (["high", "medium", "low"], {"default": "medium"}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "combine_images_to_video"
    CATEGORY = "S3/Combine"
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    def combine_images_to_video(self, images, fps, duration_per_frame, video_codec="mp4v", quality="medium"):
        
        try:
            if images is None or len(images) == 0:
                raise ValueError("At least one image is required")
            
            print(f"DEBUG: Combining {len(images)} images into video")
            print(f"DEBUG: FPS: {fps}, Duration per frame: {duration_per_frame}, Quality: {quality}")
            
            # Get workspace output directory for video file
            output_dir = os.path.join("workspace", "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate video filename
            ts = int(time.time())
            video_path = os.path.join(output_dir, f"video_{ts}.mp4")
            
            # Convert images to video using OpenCV
            try:
                import cv2
                
                # Test OpenCV installation
                print(f"DEBUG: OpenCV version: {cv2.__version__}")
                
                # Check if OpenCV was built with video support
                if not hasattr(cv2, 'VideoWriter'):
                    raise RuntimeError("OpenCV was not built with video support. Please reinstall opencv-python with: pip install opencv-python")
                
                # Get dimensions from first image
                first_img = images[0]
                if isinstance(first_img, torch.Tensor):
                    # Convert tensor to numpy
                    if first_img.dtype == torch.uint8:
                        img_np = first_img.cpu().numpy()
                    else:
                        img_np = (first_img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                    
                    # Handle batch dimension
                    if img_np.ndim == 4:
                        img_np = img_np[0]
                    
                    # Handle channel dimension
                    if img_np.ndim == 3:
                        if img_np.shape[2] in (1, 3, 4):
                            height, width = img_np.shape[0], img_np.shape[1]
                        elif img_np.shape[0] in (1, 3, 4):
                            height, width = img_np.shape[1], img_np.shape[2]
                        else:
                            raise ValueError(f"Unexpected image shape: {img_np.shape}")
                    else:
                        raise ValueError(f"Unexpected image shape: {img_np.shape}")
                else:
                    raise ValueError(f"Unsupported image type: {type(first_img)}")
                
                # Set quality parameters
                quality_params = {
                    "high": {"width": width, "height": height},
                    "medium": {"width": width//2, "height": height//2},
                    "low": {"width": width//4, "height": height//4}
                }
                
                video_width = quality_params[quality]["width"]
                video_height = quality_params[quality]["height"]
                
                # Ensure dimensions are valid (must be even numbers for most codecs)
                if video_width % 2 != 0:
                    video_width -= 1
                if video_height % 2 != 0:
                    video_height -= 1
                
                # Ensure minimum dimensions
                video_width = max(32, video_width)
                video_height = max(32, video_height)
                
                print(f"DEBUG: Video dimensions: {video_width}x{video_height}")
                
                # Try different codecs in order of preference
                codecs_to_try = [video_codec, "mp4v", "XVID", "MJPG", "X264"]
                out = None
                
                for codec in codecs_to_try:
                    try:
                        print(f"DEBUG: Trying codec: {codec}")
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        out = cv2.VideoWriter(video_path, fourcc, fps, (video_width, video_height))
                        
                        if out.isOpened():
                            print(f"DEBUG: Successfully opened video writer with codec: {codec}")
                            break
                        else:
                            print(f"DEBUG: Failed to open video writer with codec: {codec}")
                            out.release()
                            out = None
                    except Exception as e:
                        print(f"DEBUG: Error with codec {codec}: {e}")
                        if out:
                            out.release()
                            out = None
                
                if out is None or not out.isOpened():
                    # Final fallback: try with different file extension
                    video_path_avi = video_path.replace('.mp4', '.avi')
                    print(f"DEBUG: Trying fallback with AVI format: {video_path_avi}")
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(video_path_avi, fourcc, fps, (video_width, video_height))
                    
                    if out.isOpened():
                        video_path = video_path_avi  # Update path to AVI
                        print(f"DEBUG: Successfully opened AVI video writer")
                    else:
                        raise RuntimeError(f"Failed to open video writer with any codec. Tried: {codecs_to_try}")
                
                # Calculate frames per image based on duration
                frames_per_image = max(1, int(fps * duration_per_frame))
                
                # Process each image
                for img_idx, img in enumerate(images):
                    print(f"DEBUG: Processing image {img_idx + 1}/{len(images)}")
                    
                    if isinstance(img, torch.Tensor):
                        # Convert tensor to numpy
                        if img.dtype == torch.uint8:
                            img_np = img.cpu().numpy()
                        else:
                            img_np = (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
                        
                        # Handle batch dimension
                        if img_np.ndim == 4:
                            img_np = img_np[0]
                        
                        # Handle channel dimension and convert to BGR for OpenCV
                        if img_np.ndim == 3:
                            if img_np.shape[2] in (1, 3, 4):
                                # RGB format
                                if img_np.shape[2] == 3:
                                    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                                elif img_np.shape[2] == 4:
                                    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                                else:  # grayscale
                                    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                            elif img_np.shape[0] in (1, 3, 4):
                                # Channel first format, transpose
                                img_np = img_np.transpose(1, 2, 0)
                                if img_np.shape[2] == 3:
                                    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                                elif img_np.shape[2] == 4:
                                    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                                else:  # grayscale
                                    frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
                            else:
                                raise ValueError(f"Unexpected image shape: {img_np.shape}")
                        else:
                            raise ValueError(f"Unexpected image shape: {img_np.shape}")
                        
                        # Resize frame if needed
                        if frame_bgr.shape[:2] != (video_height, video_width):
                            frame_bgr = cv2.resize(frame_bgr, (video_width, video_height))
                        
                        # Write frame multiple times based on duration
                        for _ in range(frames_per_image):
                            out.write(frame_bgr)
                    else:
                        raise ValueError(f"Unsupported image type: {type(img)}")
                
                out.release()
                print(f"DEBUG: Successfully created video: {video_path}")
                
            except ImportError:
                raise RuntimeError("OpenCV (cv2) is required for video creation. Please install it with: pip install opencv-python")
            
            # Return relative path from ComfyUI root (without workspace prefix)
            relative_path = os.path.relpath(video_path, os.getcwd())
            return (relative_path,)
            
        except Exception as e:
            print(f"ERROR in S3CombineImagesToVideo: {str(e)}")
            raise e


class MultiLineTextInput:
    """
    Multi-line text input node that allows adding multiple lines of text with labels.
    Can output each line separately or concatenate them with a delimiter.
    Supports up to 10 lines for longer concatenation lists.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "line_1": ("STRING", {"default": "", "multiline": True}),
                "label_1": ("STRING", {"default": "Line 1"}),
                "line_2": ("STRING", {"default": "", "multiline": True}),
                "label_2": ("STRING", {"default": "Line 2"}),
                "line_3": ("STRING", {"default": "", "multiline": True}),
                "label_3": ("STRING", {"default": "Line 3"}),
                "line_4": ("STRING", {"default": "", "multiline": True}),
                "label_4": ("STRING", {"default": "Line 4"}),
                "line_5": ("STRING", {"default": "", "multiline": True}),
                "label_5": ("STRING", {"default": "Line 5"}),
                "line_6": ("STRING", {"default": "", "multiline": True}),
                "label_6": ("STRING", {"default": "Line 6"}),
                "line_7": ("STRING", {"default": "", "multiline": True}),
                "label_7": ("STRING", {"default": "Line 7"}),
                "line_8": ("STRING", {"default": "", "multiline": True}),
                "label_8": ("STRING", {"default": "Line 8"}),
                "line_9": ("STRING", {"default": "", "multiline": True}),
                "label_9": ("STRING", {"default": "Line 9"}),
                "line_10": ("STRING", {"default": "", "multiline": True}),
                "label_10": ("STRING", {"default": "Line 10"}),
                "output_mode": (["concatenate", "separate"], {"default": "concatenate"}),
                "delimiter": ("STRING", {"default": "\\n"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("line_1", "line_2", "line_3", "line_4", "line_5", "line_6", "line_7", "line_8", "line_9", "line_10", "concatenated")
    FUNCTION = "process_text"
    CATEGORY = "S3/Input"
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True
    
    def process_text(self, line_1, label_1, line_2, label_2, line_3, label_3, line_4, label_4, line_5, label_5,
                    line_6, label_6, line_7, label_7, line_8, label_8, line_9, label_9, line_10, label_10,
                    output_mode, delimiter):
        lines = [line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10]
        
        # Filter out empty lines for concatenation
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Process delimiter (handle escape sequences)
        processed_delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
        
        # Create concatenated output
        concatenated = processed_delimiter.join(non_empty_lines)
        
        return (line_1, line_2, line_3, line_4, line_5, line_6, line_7, line_8, line_9, line_10, concatenated)


# Registry
NODE_CLASS_MAPPINGS = {
    "S3UploadMedia": S3UploadMedia,
    "S3FileExplorer": S3FileExplorer,
    "S3SaveImageExportPath": S3SaveImageExportPath,
    "S3SaveVideoExportPath": S3SaveVideoExportPath,
    "S3SaveAudioExportPath": S3SaveAudioExportPath,
    "S3CombineImagesToVideo": S3CombineImagesToVideo,
    "MultiLineTextInput": MultiLineTextInput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "S3UploadMedia": "S3 Upload File",
    "S3FileExplorer": "File Explorer",
    "S3SaveImageExportPath": "Save Image Export Path",
    "S3SaveVideoExportPath": "Save Video Export Path",
    "S3SaveAudioExportPath": "Save Audio Export Path",
    "S3CombineImagesToVideo": "S3 Combine Images to Video",
    "MultiLineTextInput": "Multi-Line Text Input",
}
