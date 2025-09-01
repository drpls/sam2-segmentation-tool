import sys
import os
import json
from pathlib import Path
import numpy as np
from typing import List, Tuple, Optional, Dict
import cv2
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QSplitter,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QMessageBox, QProgressBar, QToolBar, QAction, QStatusBar,
    QSlider, QSpinBox, QCheckBox, QComboBox, QGroupBox,
    QShortcut, QInputDialog, QListWidgetItem, QProgressDialog
)
from PyQt5.QtCore import Qt, QPointF, pyqtSignal, QRectF, QTimer, QThread, pyqtSlot
from PyQt5.QtGui import (
    QPixmap, QPainter, QColor, QImage, QPen, QBrush, 
    QKeySequence, QCursor, QPolygonF, QTransform
)

# SAM2 imports
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    SAM2_AVAILABLE = False
    DEVICE = "cpu"
    print("SAM2 not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")


class ImageLoader(QThread):
    """Background thread for loading images"""
    imageLoaded = pyqtSignal(np.ndarray, str)
    progressUpdate = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.image_queue = []
        self.current_path = None
        
    def add_to_queue(self, path: str):
        self.image_queue.append(path)
        
    def run(self):
        while self.image_queue:
            path = self.image_queue.pop(0)
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.imageLoaded.emit(img, path)
            self.progressUpdate.emit(len(self.image_queue))


class ImageCanvas(QGraphicsView):
    """Enhanced QGraphicsView with zoom, pan, and better visualization"""
    
    clicked = pyqtSignal(QPointF, int)  # position, button (1=left, 2=right)
    mouseMoved = pyqtSignal(QPointF)
    
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.image_item = None
        self.mask_items = []
        self.current_mask_item = None
        self.saved_mask_items = []  # Store saved masks for visualization
        
        # Visual feedback
        self.click_markers = []
        self.hover_marker = None
        
        # Zoom/Pan
        self.zoom_factor = 1.0
        self.pan_mode = False
        self.last_pan_point = QPointF()
        
        # Settings
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        
        # Mask visualization settings
        self.mask_opacity = 100
        self.mask_colors = [
            QColor(255, 0, 0),    # Red
            QColor(0, 0, 255),    # Blue
            QColor(255, 255, 0),  # Yellow
            QColor(255, 0, 255),  # Magenta
            QColor(0, 255, 255),  # Cyan
            QColor(255, 128, 0),  # Orange
            QColor(128, 0, 255),  # Purple
            QColor(255, 192, 203),# Pink
        ]
        self.current_color_idx = 0
        
    def set_image(self, image_path: str):
        """Load and display an image"""
        self.scene.clear()
        self.mask_items = []
        self.saved_mask_items = []
        self.current_mask_item = None
        self.click_markers = []
        self.zoom_factor = 1.0
        
        pixmap = QPixmap(image_path)
        self.image_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(self.image_item.boundingRect())
        self.fitInView(self.image_item, Qt.KeepAspectRatio)
        
    def wheelEvent(self, event):
        """Zoom with mouse wheel"""
        if event.modifiers() == Qt.ControlModifier:
            # Zoom in/out with Ctrl+Wheel
            scale_factor = 1.15
            if event.angleDelta().y() > 0:
                self.zoom_factor *= scale_factor
                self.scale(scale_factor, scale_factor)
            else:
                self.zoom_factor /= scale_factor
                self.scale(1/scale_factor, 1/scale_factor)
        else:
            super().wheelEvent(event)
            
    def mousePressEvent(self, event):
        """Handle mouse clicks for segmentation and pan"""
        if event.button() == Qt.MiddleButton:
            # Start panning
            self.pan_mode = True
            self.last_pan_point = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        elif self.image_item:
            scene_pos = self.mapToScene(event.pos())
            
            # Check if click is within image bounds
            if self.image_item.boundingRect().contains(scene_pos):
                if event.modifiers() != Qt.ShiftModifier:  # Normal click for segmentation
                    button = 1 if event.button() == Qt.LeftButton else 2
                    self.clicked.emit(scene_pos, button)
                    
                    # Add visual feedback
                    color = QColor(0, 255, 0, 200) if button == 1 else QColor(255, 0, 0, 200)
                    self.add_click_marker(scene_pos, color, button == 1)
                    
        super().mousePressEvent(event)
        
    def mouseMoveEvent(self, event):
        """Handle mouse movement for hover effects and panning"""
        if self.pan_mode:
            # Pan the view
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            
            # Adjust scrollbars
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            # Emit mouse position for coordinate display
            if self.image_item:
                scene_pos = self.mapToScene(event.pos())
                if self.image_item.boundingRect().contains(scene_pos):
                    self.mouseMoved.emit(scene_pos)
                    self.update_hover_marker(scene_pos)
                    
        super().mouseMoveEvent(event)
        
    def mouseReleaseEvent(self, event):
        """End panning"""
        if event.button() == Qt.MiddleButton:
            self.pan_mode = False
            self.setCursor(QCursor(Qt.ArrowCursor))
        super().mouseReleaseEvent(event)
        
    def add_click_marker(self, pos: QPointF, color: QColor, is_positive: bool):
        """Add enhanced visual marker for click position"""
        # Draw circle
        pen = QPen(color, 2)
        brush = QBrush(color)
        marker = self.scene.addEllipse(pos.x()-4, pos.y()-4, 8, 8, pen, brush)
        
        # Draw + or - symbol
        text_color = QColor(255, 255, 255)
        symbol = "+" if is_positive else "-"
        text = self.scene.addText(symbol)
        text.setDefaultTextColor(text_color)
        text.setPos(pos.x() - 5, pos.y() - 10)
        text.setScale(0.8)
        
        self.click_markers.append((marker, text))
        
    def update_hover_marker(self, pos: QPointF):
        """Show hover marker for better precision"""
        if self.hover_marker:
            self.scene.removeItem(self.hover_marker)
            
        pen = QPen(QColor(255, 255, 255, 100), 1, Qt.DashLine)
        self.hover_marker = self.scene.addEllipse(
            pos.x()-10, pos.y()-10, 20, 20, pen
        )
        
    def display_mask(self, mask: np.ndarray, is_saved: bool = False):
        """Display mask with better visualization"""
        if not is_saved and self.current_mask_item:
            self.scene.removeItem(self.current_mask_item)
            
        # Get color for mask
        if is_saved:
            # Usa i colori dell'array per le maschere salvate (NO verde)
            color = self.mask_colors[self.current_color_idx % len(self.mask_colors)]
            self.current_color_idx += 1
        else:
            # Verde SOLO per la maschera in creazione
            color = QColor(0, 255, 0)
            
        color.setAlpha(self.mask_opacity)
        
        # Create optimized mask image
        h, w = mask.shape
        mask_image = QImage(w, h, QImage.Format_ARGB32)
        mask_image.fill(Qt.transparent)
        
        # Use faster numpy operations
        mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        mask_rgba[:,:,0] = color.red()
        mask_rgba[:,:,1] = color.green()
        mask_rgba[:,:,2] = color.blue()
        mask_rgba[:,:,3] = (mask * color.alpha()).astype(np.uint8)
        
        # Convert to QImage efficiently
        mask_image = QImage(mask_rgba.data, w, h, w*4, QImage.Format_ARGB32)
        mask_pixmap = QPixmap.fromImage(mask_image)
        
        if is_saved:
            item = self.scene.addPixmap(mask_pixmap)
            self.saved_mask_items.append(item)
        else:
            self.current_mask_item = self.scene.addPixmap(mask_pixmap)
            
    def set_mask_opacity(self, opacity: int):
        """Update mask opacity"""
        self.mask_opacity = opacity
        # Refresh current mask if exists
        if self.current_mask_item:
            # Would need to store and redraw the mask
            pass
            
    def clear_current_mask(self):
        """Clear current mask and markers"""
        if self.current_mask_item:
            self.scene.removeItem(self.current_mask_item)
            self.current_mask_item = None
        
        # Clear click markers
        for marker, text in self.click_markers:
            self.scene.removeItem(marker)
            self.scene.removeItem(text)
        self.click_markers = []
        
        if self.hover_marker:
            self.scene.removeItem(self.hover_marker)
            self.hover_marker = None
            
    def clear_all_masks(self):
        """Clear all saved masks from display"""
        for item in self.saved_mask_items:
            self.scene.removeItem(item)
        self.saved_mask_items = []
        self.current_color_idx = 0
        
    def reset_view(self):
        """Reset zoom and center view"""
        if self.image_item:
            self.resetTransform()
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
            self.zoom_factor = 1.0
            
    def resizeEvent(self, event):
        """Maintain aspect ratio when resizing"""
        super().resizeEvent(event)
        if self.image_item and self.zoom_factor == 1.0:
            self.fitInView(self.image_item, Qt.KeepAspectRatio)


class MaskManager:
    """Enhanced mask manager with undo/redo and metadata"""
    
    def __init__(self):
        self.masks = []  # List of (mask, metadata) tuples
        self.current_image_shape = None
        self.history = deque(maxlen=20)  # Undo history
        self.redo_stack = []
        
    def add_mask(self, mask: np.ndarray, metadata: Dict = None):
        """Add mask with metadata"""
        if metadata is None:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'clicks': 0
            }
        
        # Save current state for undo
        self.history.append(list(self.masks))
        self.redo_stack.clear()
        
        self.masks.append((mask.copy(), metadata))
        
    def remove_mask(self, index: int):
        """Remove a specific mask"""
        if 0 <= index < len(self.masks):
            self.history.append(list(self.masks))
            self.redo_stack.clear()
            del self.masks[index]
            
    def undo(self):
        """Undo last action"""
        if self.history:
            self.redo_stack.append(list(self.masks))
            self.masks = self.history.pop()
            return True
        return False
        
    def redo(self):
        """Redo last undone action"""
        if self.redo_stack:
            self.history.append(list(self.masks))
            self.masks = self.redo_stack.pop()
            return True
        return False
        
    def get_combined_mask(self) -> Optional[np.ndarray]:
        """Get combined mask of all saved masks"""
        if not self.masks:
            return None
            
        combined = np.zeros_like(self.masks[0][0], dtype=bool)
        for mask, _ in self.masks:
            combined = np.logical_or(combined, mask)
        return combined
        
    def get_background_mask(self) -> Optional[np.ndarray]:
        """Get mask of areas not covered by any mask"""
        if not self.masks or self.current_image_shape is None:
            return None
            
        combined = self.get_combined_mask()
        return np.logical_not(combined)
        
    def clear(self):
        """Clear all masks and history"""
        self.masks = []
        self.history.clear()
        self.redo_stack.clear()
        
    def set_image_shape(self, shape: Tuple[int, int]):
        """Set the shape of the current image"""
        self.current_image_shape = shape
        
    def export_metadata(self) -> Dict:
        """Export masks metadata"""
        return {
            'image_shape': self.current_image_shape,
            'masks': [metadata for _, metadata in self.masks],
            'total_masks': len(self.masks)
        }


class SAM2SegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM2 Interactive Segmentation Tool - Enhanced")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Initialize SAM2
        self.predictor = None
        self.init_sam2()
        
        # Image loader thread
        self.image_loader = ImageLoader()
        self.image_loader.imageLoaded.connect(self.on_image_loaded)
        
        # State variables
        self.current_image_path = None
        self.current_image = None
        self.image_files = []
        self.current_image_index = 0
        self.mask_manager = MaskManager()
        
        # Current segmentation state
        self.current_points = []
        self.current_labels = []
        self.current_mask = None
        self.auto_predict = True
        self.prediction_cache = {}
        
        # Project management
        self.project_dir = None
        self.auto_save = False
        
        # Setup UI
        self.setup_ui()
        self.setup_toolbar()
        self.setup_statusbar()
        self.setup_shortcuts()
        
    def init_sam2(self):
        """Initialize SAM2 model with error handling"""
        if not SAM2_AVAILABLE:
            QMessageBox.warning(self, "Warning", 
                "SAM2 is not installed. Please install it first.\n"
                "pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            return
            
        try:
            model_cfg = "sam2_hiera_l"  # SENZA .yaml e SENZA percorso
            checkpoint = "checkpoints/sam2_hiera_large.pt"
            
            if not os.path.exists(checkpoint):
                QMessageBox.warning(self, "Model Not Found",
                    f"SAM2 checkpoint not found at {checkpoint}\n"
                    f"Download it with:\n"
                    f"wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O {checkpoint}")
                return
            
            print(f"Loading SAM2 from {checkpoint} on {DEVICE}...")
            
            # Ordine corretto: config_name (stringa), checkpoint_path
            model = build_sam2(model_cfg, checkpoint, device=DEVICE)
            self.predictor = SAM2ImagePredictor(model)
            print("SAM2 loaded successfully!")
            
        except Exception as e:
            print(f"Error initializing SAM2: {e}")
            QMessageBox.warning(self, "Initialization Error", 
                f"Failed to initialize SAM2:\n{str(e)}")
            
    def setup_ui(self):
        """Setup enhanced UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Controls
        left_panel = self.create_left_panel()
        
        # Center - Canvas
        self.canvas = ImageCanvas()
        self.canvas.clicked.connect(self.on_canvas_click)
        self.canvas.mouseMoved.connect(self.on_mouse_move)
        
        # Right panel - Properties
        right_panel = self.create_right_panel()
        
        # Add to layout with splitters
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 3)  # Canvas gets more space
        splitter.setSizes([300, 900, 300])
        
        main_layout.addWidget(splitter)
        
    def create_left_panel(self):
        """Create left control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        panel.setMaximumWidth(350)
        
        # Project controls
        project_group = QGroupBox("Project")
        project_layout = QVBoxLayout()
        
        self.load_folder_btn = QPushButton("Load Image Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        project_layout.addWidget(self.load_folder_btn)
        
        self.save_project_btn = QPushButton("Save Project")
        self.save_project_btn.clicked.connect(self.save_project)
        project_layout.addWidget(self.save_project_btn)
        
        self.load_project_btn = QPushButton("Load Project")
        self.load_project_btn.clicked.connect(self.load_project)
        project_layout.addWidget(self.load_project_btn)
        
        project_group.setLayout(project_layout)
        layout.addWidget(project_group)
        
        # Image list
        layout.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.on_image_selected)
        layout.addWidget(self.image_list)
        
        # Mask controls
        mask_group = QGroupBox("Mask Operations")
        mask_layout = QVBoxLayout()
        
        self.save_mask_btn = QPushButton("Save Current Mask (S)")
        self.save_mask_btn.clicked.connect(self.save_current_mask)
        self.save_mask_btn.setEnabled(False)
        mask_layout.addWidget(self.save_mask_btn)
        
        self.clear_mask_btn = QPushButton("Clear Current (C)")
        self.clear_mask_btn.clicked.connect(self.clear_current_mask)
        mask_layout.addWidget(self.clear_mask_btn)
        
        # Undo/Redo
        undo_redo_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.undo_action)
        undo_redo_layout.addWidget(self.undo_btn)
        
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.redo_action)
        undo_redo_layout.addWidget(self.redo_btn)
        mask_layout.addLayout(undo_redo_layout)
        
        mask_group.setLayout(mask_layout)
        layout.addWidget(mask_group)
        
        # Saved masks list
        layout.addWidget(QLabel("Saved Masks:"))
        self.saved_masks_list = QListWidget()
        self.saved_masks_list.itemDoubleClicked.connect(self.on_mask_double_clicked)
        layout.addWidget(self.saved_masks_list)
        
        # Export controls
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()
        
        self.batch_export_btn = QPushButton("Batch Export All")
        self.batch_export_btn.clicked.connect(self.batch_export)
        export_layout.addWidget(self.batch_export_btn)
        
        self.export_format = QComboBox()
        self.export_format.addItems(["PNG (Transparent)", "JPG (Mask)", "Both"])
        export_layout.addWidget(self.export_format)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("< Previous")
        self.prev_btn.clicked.connect(self.prev_image)
        nav_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("Next >")
        self.next_btn.clicked.connect(self.next_image)
        nav_layout.addWidget(self.next_btn)
        
        layout.addLayout(nav_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        return panel
        
    def create_right_panel(self):
        """Create right properties panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        panel.setMaximumWidth(350)
        
        # View controls
        view_group = QGroupBox("View Controls")
        view_layout = QVBoxLayout()
        
        # Opacity slider
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Mask Opacity:"))
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 255)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)
        opacity_layout.addWidget(self.opacity_slider)
        
        self.opacity_label = QLabel("100")
        opacity_layout.addWidget(self.opacity_label)
        view_layout.addLayout(opacity_layout)
        
        # Show/Hide options
        self.show_masks_cb = QCheckBox("Show Saved Masks")
        self.show_masks_cb.setChecked(True)
        self.show_masks_cb.toggled.connect(self.toggle_saved_masks)
        view_layout.addWidget(self.show_masks_cb)
        
        self.show_markers_cb = QCheckBox("Show Click Markers")
        self.show_markers_cb.setChecked(True)
        view_layout.addWidget(self.show_markers_cb)
        
        # Reset view button
        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.clicked.connect(lambda: self.canvas.reset_view())
        view_layout.addWidget(self.reset_view_btn)
        
        view_group.setLayout(view_layout)
        layout.addWidget(view_group)
        
        # SAM Settings
        sam_group = QGroupBox("SAM2 Settings")
        sam_layout = QVBoxLayout()
        
        self.auto_predict_cb = QCheckBox("Auto Predict")
        self.auto_predict_cb.setChecked(True)
        self.auto_predict_cb.toggled.connect(self.toggle_auto_predict)
        sam_layout.addWidget(self.auto_predict_cb)
        
        self.multimask_cb = QCheckBox("Multi-mask Output")
        self.multimask_cb.setChecked(True)
        sam_layout.addWidget(self.multimask_cb)
        
        # Points per side for auto-segmentation
        points_layout = QHBoxLayout()
        points_layout.addWidget(QLabel("Points/Side:"))
        self.points_spinbox = QSpinBox()
        self.points_spinbox.setRange(4, 64)
        self.points_spinbox.setValue(32)
        points_layout.addWidget(self.points_spinbox)
        sam_layout.addLayout(points_layout)
        
        self.auto_segment_btn = QPushButton("Auto Segment All")
        self.auto_segment_btn.clicked.connect(self.auto_segment_image)
        sam_layout.addWidget(self.auto_segment_btn)
        
        sam_group.setLayout(sam_layout)
        layout.addWidget(sam_group)
        
        # Statistics
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout()
        
        self.stats_label = QLabel("No image loaded")
        stats_layout.addWidget(self.stats_label)
        
        self.coords_label = QLabel("X: 0, Y: 0")
        stats_layout.addWidget(self.coords_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Auto-save
        self.auto_save_cb = QCheckBox("Auto-save Masks")
        self.auto_save_cb.toggled.connect(self.toggle_auto_save)
        layout.addWidget(self.auto_save_cb)
        
        layout.addStretch()
        return panel
        
    def setup_toolbar(self):
        """Setup enhanced toolbar"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # File operations
        toolbar.addAction("Open", self.load_folder)
        toolbar.addAction("Save", self.save_project)
        toolbar.addSeparator()
        
        # Mask operations
        toolbar.addAction("Save Mask", self.save_current_mask)
        toolbar.addAction("Clear", self.clear_current_mask)
        toolbar.addAction("↶ Undo", self.undo_action)
        toolbar.addAction("↷ Redo", self.redo_action)
        toolbar.addSeparator()
        
        # View operations
        toolbar.addAction("+ Zoom In", lambda: self.canvas.scale(1.2, 1.2))
        toolbar.addAction("- Zoom Out", lambda: self.canvas.scale(0.8, 0.8))
        toolbar.addAction("Reset", lambda: self.canvas.reset_view())
        toolbar.addSeparator()
        
        # Help
        toolbar.addAction("Help", self.show_help)
        toolbar.addAction("About", self.show_about)
        
    def setup_statusbar(self):
        """Setup enhanced status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Add permanent widgets
        self.device_label = QLabel(f"Device: {DEVICE}")
        self.statusbar.addPermanentWidget(self.device_label)
        
        self.image_count_label = QLabel("Images: 0/0")
        self.statusbar.addPermanentWidget(self.image_count_label)
        
        self.statusbar.showMessage("Ready. Load a folder to begin.")
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        shortcuts = {
            'Ctrl+O': self.load_folder,
            'Ctrl+S': self.save_project,
            'S': self.save_current_mask,
            'C': self.clear_current_mask,
            'Ctrl+Z': self.undo_action,
            'Ctrl+Y': self.redo_action,
            'Space': self.next_image,
            'Backspace': self.prev_image,
            'Ctrl+E': self.batch_export,
            'R': lambda: self.canvas.reset_view(),
            'H': self.toggle_saved_masks,
            'Ctrl+Q': self.close,
            'F1': self.show_help,
            'Del': self.delete_selected_mask,
        }
        
        for key, func in shortcuts.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(func)
            
    def load_folder(self):
        """Load folder with progress indication"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", 
            self.project_dir or ""
        )
        
        if folder_path:
            self.project_dir = folder_path
            
            # Find all image files
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
            self.image_files = []
            
            for ext in image_extensions:
                self.image_files.extend(Path(folder_path).glob(f'*{ext}'))
                self.image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
                
            self.image_files = sorted(self.image_files)
            
            if self.image_files:
                # Update UI
                self.image_list.clear()
                for img_path in self.image_files:
                    item = QListWidgetItem(img_path.name)
                    # Check if masks exist for this image
                    mask_file = img_path.parent / f".masks/{img_path.stem}_masks.npz"
                    if mask_file.exists():
                        item.setText(f"✓ {img_path.name}")
                    self.image_list.addItem(item)
                    
                # Load first image
                self.current_image_index = 0
                self.image_list.setCurrentRow(0)
                self.update_navigation_buttons()
                self.update_image_count()
                
                self.statusbar.showMessage(f"Loaded {len(self.image_files)} images from {Path(folder_path).name}")
            else:
                QMessageBox.warning(self, "No Images", "No images found in the selected folder")
                
    @pyqtSlot(np.ndarray, str)
    def on_image_loaded(self, image: np.ndarray, path: str):
        """Handle asynchronously loaded image"""
        if path == self.current_image_path:
            self.current_image = image
            if self.predictor:
                self.predictor.set_image(self.current_image)
                
    def on_image_selected(self, index):
        """Handle image selection"""
        if 0 <= index < len(self.image_files):
            # Save current masks if auto-save is on
            if self.auto_save and self.mask_manager.masks:
                self.auto_save_masks()
                
            self.current_image_index = index
            self.load_image(self.image_files[index])
            self.update_navigation_buttons()
            self.update_image_count()
            
    def load_image(self, image_path: Path):
        """Load and display image with caching"""
        self.current_image_path = str(image_path)
        
        # Check cache
        if self.current_image_path in self.prediction_cache:
            self.current_image = self.prediction_cache[self.current_image_path]['image']
        else:
            self.current_image = cv2.imread(self.current_image_path)
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
        # Update canvas
        self.canvas.set_image(self.current_image_path)
        
        # Reset mask manager
        self.mask_manager.clear()
        self.mask_manager.set_image_shape(self.current_image.shape[:2])
        
        # Load existing masks if any
        self.load_existing_masks(image_path)
        
        # Clear current segmentation
        self.clear_current_mask()
        
        # Set image for SAM2
        if self.predictor:
            self.predictor.set_image(self.current_image)
            
        # Update UI
        self.update_stats()
        self.statusbar.showMessage(f"Loaded: {image_path.name}")
        
    def load_existing_masks(self, image_path: Path):
        """Load existing masks for the image"""
        mask_file = image_path.parent / f".masks/{image_path.stem}_masks.npz"
        
        if mask_file.exists():
            try:
                data = np.load(mask_file, allow_pickle=True)
                masks = data['masks']
                metadata = data.get('metadata', [{}] * len(masks))
                
                self.saved_masks_list.clear()
                for i, (mask, meta) in enumerate(zip(masks, metadata)):
                    self.mask_manager.masks.append((mask, dict(meta)))
                    self.saved_masks_list.addItem(f"Mask {i+1}")
                    self.canvas.display_mask(mask, is_saved=True)
                    
                self.statusbar.showMessage(f"Loaded {len(masks)} existing masks")
            except Exception as e:
                print(f"Error loading masks: {e}")
                
    def on_canvas_click(self, pos: QPointF, button: int):
        """Handle canvas click with caching"""
        if not self.current_image_path or not self.predictor:
            return
            
        # Convert to image coordinates
        x, y = int(pos.x()), int(pos.y())
        
        # Add point
        self.current_points.append([x, y])
        self.current_labels.append(1 if button == 1 else 0)
        
        # Run segmentation if auto-predict is on
        if self.auto_predict:
            self.run_segmentation()
            
    def on_mouse_move(self, pos: QPointF):
        """Update coordinate display"""
        x, y = int(pos.x()), int(pos.y())
        self.coords_label.setText(f"X: {x}, Y: {y}")
        
    def run_segmentation(self):
        """Run SAM2 segmentation with caching"""
        if not self.predictor or not self.current_points:
            return
            
        try:
            # Create cache key
            cache_key = f"{self.current_image_path}_{self.current_points}_{self.current_labels}"
            
            if cache_key in self.prediction_cache:
                # Use cached result
                masks, scores = self.prediction_cache[cache_key]
            else:
                # Run prediction
                input_points = np.array(self.current_points)
                input_labels = np.array(self.current_labels)
                
                masks, scores, _ = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=self.multimask_cb.isChecked()
                )
                
                # Cache result (limit cache size)
                if len(self.prediction_cache) > 100:
                    self.prediction_cache.clear()
                self.prediction_cache[cache_key] = (masks, scores)
            
            # Select best mask
            best_mask_idx = np.argmax(scores)
            self.current_mask = masks[best_mask_idx]
            
            # Display mask
            self.canvas.display_mask(self.current_mask)
            
            # Enable save button
            self.save_mask_btn.setEnabled(True)
            
            self.statusbar.showMessage(
                f"Segmentation updated. Score: {scores[best_mask_idx]:.3f} | "
                f"Points: {len(self.current_points)}"
            )
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            QMessageBox.warning(self, "Segmentation Error", f"Failed: {str(e)}")
            
    def save_current_mask(self):
        """Save current mask with metadata"""
        if self.current_mask is not None:
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'clicks': len(self.current_points),
                'points': self.current_points,
                'labels': self.current_labels
            }
            
            self.mask_manager.add_mask(self.current_mask, metadata)
            
            # Update UI
            mask_count = len(self.mask_manager.masks)
            item = QListWidgetItem(f"Mask {mask_count}")
            self.saved_masks_list.addItem(item)
            
            # Display as saved mask
            self.canvas.display_mask(self.current_mask, is_saved=True)
            
            # Clear for next mask
            self.clear_current_mask()
            
            # Auto-save if enabled
            if self.auto_save:
                self.auto_save_masks()
                
            self.update_stats()
            self.statusbar.showMessage(f"Mask saved. Total: {mask_count}")
            
    def clear_current_mask(self):
        """Clear current mask"""
        self.current_points = []
        self.current_labels = []
        self.current_mask = None
        
        self.canvas.clear_current_mask()
        self.save_mask_btn.setEnabled(False)
        
        self.statusbar.showMessage("Current mask cleared")
        
    def delete_selected_mask(self):
        """Delete selected saved mask"""
        current_row = self.saved_masks_list.currentRow()
        if current_row >= 0:
            self.mask_manager.remove_mask(current_row)
            self.saved_masks_list.takeItem(current_row)
            
            # Refresh display
            self.refresh_mask_display()
            self.update_stats()
            
    def refresh_mask_display(self):
        """Refresh all masks on canvas"""
        self.canvas.clear_all_masks()
        for mask, _ in self.mask_manager.masks:
            self.canvas.display_mask(mask, is_saved=True)
            
    def on_mask_double_clicked(self, item):
        """Handle double-click on mask item"""
        # Could implement mask editing or detailed view
        row = self.saved_masks_list.row(item)
        if 0 <= row < len(self.mask_manager.masks):
            mask, metadata = self.mask_manager.masks[row]
            QMessageBox.information(self, "Mask Info", 
                f"Mask {row+1}\n"
                f"Created: {metadata.get('timestamp', 'Unknown')}\n"
                f"Clicks: {metadata.get('clicks', 0)}")
                
    def undo_action(self):
        """Undo last mask action"""
        if self.mask_manager.undo():
            self.refresh_ui_after_undo_redo()
            self.statusbar.showMessage("Undone")
            
    def redo_action(self):
        """Redo last undone action"""
        if self.mask_manager.redo():
            self.refresh_ui_after_undo_redo()
            self.statusbar.showMessage("Redone")
            
    def refresh_ui_after_undo_redo(self):
        """Refresh UI after undo/redo"""
        self.saved_masks_list.clear()
        for i, (mask, _) in enumerate(self.mask_manager.masks):
            self.saved_masks_list.addItem(f"Mask {i+1}")
        self.refresh_mask_display()
        self.update_stats()
        
    def toggle_auto_predict(self, checked):
        """Toggle auto prediction"""
        self.auto_predict = checked
        if not checked and self.current_points:
            # Run prediction manually
            self.run_segmentation()
            
    def toggle_saved_masks(self):
        """Toggle visibility of saved masks"""
        checked = self.show_masks_cb.isChecked()
        for item in self.canvas.saved_mask_items:
            item.setVisible(checked)
            
    def on_opacity_changed(self, value):
        """Update mask opacity"""
        self.opacity_label.setText(str(value))
        self.canvas.set_mask_opacity(value)
        self.refresh_mask_display()
        
    def auto_segment_image(self):
        """Auto-segment entire image"""
        if not self.predictor or not self.current_image_path:
            return
            
        try:
            # Show progress dialog
            progress = QProgressDialog("Auto-segmenting image...", "Cancel", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            # Generate automatic masks
            # This is a simplified version - SAM2 has more sophisticated methods
            points_per_side = self.points_spinbox.value()
            h, w = self.current_image.shape[:2]
            
            x_points = np.linspace(0, w, points_per_side)
            y_points = np.linspace(0, h, points_per_side)
            
            masks_found = 0
            total_points = points_per_side * points_per_side
            
            for i, x in enumerate(x_points):
                for j, y in enumerate(y_points):
                    if progress.wasCanceled():
                        break
                        
                    progress.setValue(int((i * points_per_side + j) / total_points * 100))
                    
                    # Try to segment at this point
                    point = np.array([[x, y]])
                    label = np.array([1])
                    
                    masks, scores, _ = self.predictor.predict(
                        point_coords=point,
                        point_labels=label,
                        multimask_output=False
                    )
                    
                    if scores[0] > 0.9:  # High confidence threshold
                        self.mask_manager.add_mask(masks[0], {
                            'auto_generated': True,
                            'score': float(scores[0])
                        })
                        masks_found += 1
                        
            progress.close()
            
            # Update UI
            self.refresh_ui_after_undo_redo()
            QMessageBox.information(self, "Auto-segmentation Complete", 
                f"Found {masks_found} masks")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Auto-segmentation failed: {e}")
            
    def batch_export(self):
        """Batch export all images with masks"""
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "Please load a folder first")
            return
            
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.project_dir
        )
        
        if not output_dir:
            return
            
        output_path = Path(output_dir)
        format_option = self.export_format.currentText()
        
        # Create subdirectories
        if "PNG" in format_option or format_option == "Both":
            (output_path / "png_masks").mkdir(exist_ok=True)
        if "JPG" in format_option or format_option == "Both":
            (output_path / "jpg_masks").mkdir(exist_ok=True)
        (output_path / "backgrounds").mkdir(exist_ok=True)
        
        # IMPORTANTE: Prima salva le maschere correnti se ci sono
        if self.mask_manager.masks and self.current_image_path:
            self.auto_save_masks()
        
        # Progress dialog
        progress = QProgressDialog("Exporting masks...", "Cancel", 0, len(self.image_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        exported_count = 0
        
        for idx, img_path in enumerate(self.image_files):
            if progress.wasCanceled():
                break
                
            progress.setValue(idx)
            progress.setLabelText(f"Processing {img_path.name}...")
            
            # Load masks for this image
            mask_file = img_path.parent / f".masks/{img_path.stem}_masks.npz"
            
            if mask_file.exists():
                try:
                    data = np.load(mask_file, allow_pickle=True)
                    masks = data['masks']
                    
                    # Load image
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Export each mask
                    for i, mask in enumerate(masks):
                        base_name = f"{img_path.stem}_mask_{i+1}"
                        
                        if "PNG" in format_option or format_option == "Both":
                            png_path = output_path / "png_masks" / f"{base_name}.png"
                            self._export_mask_png(img_rgb, mask, png_path)
                            
                        if "JPG" in format_option or format_option == "Both":
                            jpg_path = output_path / "jpg_masks" / f"{base_name}.jpg"
                            self._export_mask_jpg(mask, jpg_path)
                        
                        exported_count += 1
                        
                    # Export background
                    if len(masks) > 0:
                        combined = np.zeros_like(masks[0], dtype=bool)
                        for mask in masks:
                            combined = np.logical_or(combined, mask)
                        background = np.logical_not(combined)
                        bg_path = output_path / "backgrounds" / f"{img_path.stem}_bg.png"
                        self._export_mask_png(img_rgb, background, bg_path)
                        
                except Exception as e:
                    print(f"Error exporting {img_path.name}: {e}")
                        
        progress.close()
        
        if exported_count > 0:
            QMessageBox.information(self, "Export Complete", 
                f"Exported {exported_count} masks to {output_dir}")
        else:
            QMessageBox.warning(self, "No Masks", 
                "No masks found to export. Create and save masks first.")
            
    def _export_mask_png(self, image: np.ndarray, mask: np.ndarray, save_path: Path):
        """Export mask as PNG with transparency"""
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:,:,:3] = image
        rgba[:,:,3] = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(save_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        
    def _export_mask_jpg(self, mask: np.ndarray, save_path: Path):
        """Export mask as JPG"""
        mask_img = (mask * 255).astype(np.uint8)
        cv2.imwrite(str(save_path), mask_img)
        
    def auto_save_masks(self):
        """Auto-save masks for current image"""
        if not self.current_image_path or not self.mask_manager.masks:
            return
            
        img_path = Path(self.current_image_path)
        mask_dir = img_path.parent / ".masks"
        mask_dir.mkdir(exist_ok=True)
        
        # Save masks
        masks = np.array([m for m, _ in self.mask_manager.masks])
        metadata = [meta for _, meta in self.mask_manager.masks]
        
        np.savez_compressed(
            mask_dir / f"{img_path.stem}_masks.npz",
            masks=masks,
            metadata=metadata
        )
        
        # Update image list item
        current_item = self.image_list.item(self.current_image_index)
        if current_item and not current_item.text().startswith("✓"):
            current_item.setText(f"✓ {img_path.name}")
            
    def toggle_auto_save(self, checked):
        """Toggle auto-save feature"""
        self.auto_save = checked
        if checked:
            self.statusbar.showMessage("Auto-save enabled", 2000)
            
    def save_project(self):
        """Save entire project"""
        if not self.project_dir:
            QMessageBox.warning(self, "No Project", "No project loaded")
            return
            
        # Save current masks
        if self.mask_manager.masks:
            self.auto_save_masks()
            
        # Save project metadata
        project_file = Path(self.project_dir) / ".sam2_project.json"
        project_data = {
            'version': '1.0',
            'created': datetime.now().isoformat(),
            'total_images': len(self.image_files),
            'current_index': self.current_image_index,
            'settings': {
                'auto_save': self.auto_save,
                'opacity': self.opacity_slider.value(),
                'multimask': self.multimask_cb.isChecked(),
                'points_per_side': self.points_spinbox.value()
            }
        }
        
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2)
            
        QMessageBox.information(self, "Project Saved", 
            f"Project saved to {self.project_dir}")
            
    def load_project(self):
        """Load a saved project"""
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Project Directory"
        )
        
        if not project_dir:
            return
            
        project_file = Path(project_dir) / ".sam2_project.json"
        
        if not project_file.exists():
            QMessageBox.warning(self, "Not a Project", 
                "Selected directory is not a SAM2 project")
            return
            
        # Load project data
        with open(project_file, 'r') as f:
            project_data = json.load(f)
            
        # Apply settings
        settings = project_data.get('settings', {})
        self.auto_save_cb.setChecked(settings.get('auto_save', False))
        self.opacity_slider.setValue(settings.get('opacity', 100))
        self.multimask_cb.setChecked(settings.get('multimask', True))
        self.points_spinbox.setValue(settings.get('points_per_side', 32))
        
        # Load folder
        self.project_dir = project_dir
        self.load_folder()
        
        # Go to last position
        if self.image_files:
            index = project_data.get('current_index', 0)
            self.current_image_index = min(index, len(self.image_files) - 1)
            self.image_list.setCurrentRow(self.current_image_index)
            
    def update_stats(self):
        """Update statistics display"""
        if self.current_image is not None:
            h, w, c = self.current_image.shape
            masks = len(self.mask_manager.masks)
            coverage = 0
            
            if masks > 0:
                combined = self.mask_manager.get_combined_mask()
                coverage = (np.sum(combined) / (h * w)) * 100
                
            self.stats_label.setText(
                f"Image: {w}×{h}\n"
                f"Masks: {masks}\n"
                f"Coverage: {coverage:.1f}%"
            )
        else:
            self.stats_label.setText("No image loaded")
            
    def update_navigation_buttons(self):
        """Update navigation button states"""
        self.prev_btn.setEnabled(self.current_image_index > 0)
        self.next_btn.setEnabled(self.current_image_index < len(self.image_files) - 1)
        
    def update_image_count(self):
        """Update image count display"""
        if self.image_files:
            self.image_count_label.setText(
                f"Image: {self.current_image_index + 1}/{len(self.image_files)}"
            )
        else:
            self.image_count_label.setText("Images: 0/0")
            
    def prev_image(self):
        """Go to previous image"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.image_list.setCurrentRow(self.current_image_index)
            
    def next_image(self):
        """Go to next image"""
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.image_list.setCurrentRow(self.current_image_index)
            
    def show_help(self):
        """Show help dialog"""
        help_text = """
        <h2>SAM2 Interactive Segmentation Tool</h2>
        
        <h3>Basic Usage:</h3>
        <ul>
        <li><b>Left Click:</b> Add to mask (positive point)</li>
        <li><b>Right Click:</b> Remove from mask (negative point)</li>
        <li><b>Middle Mouse:</b> Pan the view</li>
        <li><b>Ctrl+Wheel:</b> Zoom in/out</li>
        </ul>
        
        <h3>Keyboard Shortcuts:</h3>
        <ul>
        <li><b>S:</b> Save current mask</li>
        <li><b>C:</b> Clear current mask</li>
        <li><b>Ctrl+Z:</b> Undo</li>
        <li><b>Ctrl+Y:</b> Redo</li>
        <li><b>Space:</b> Next image</li>
        <li><b>Backspace:</b> Previous image</li>
        <li><b>Del:</b> Delete selected mask</li>
        <li><b>R:</b> Reset view</li>
        <li><b>H:</b> Toggle mask visibility</li>
        <li><b>Ctrl+E:</b> Batch export</li>
        <li><b>F1:</b> Show this help</li>
        </ul>
        
        <h3>Features:</h3>
        <ul>
        <li>Auto-save masks</li>
        <li>Batch export to PNG/JPG</li>
        <li>Project management</li>
        <li>Undo/Redo support</li>
        <li>Auto-segmentation</li>
        <li>Multi-mask support per image</li>
        </ul>
        """
        
        msg = QMessageBox()
        msg.setWindowTitle("Help")
        msg.setTextFormat(Qt.RichText)
        msg.setText(help_text)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
        
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About",
            "SAM2 Interactive Segmentation Tool\n"
            "Version 2.0 Enhanced\n\n"
            "Built with PyQt5 and SAM2\n"
            "Enhanced with advanced features")
            
    def closeEvent(self, event):
        """Handle application close"""
        # Save current work if auto-save is on
        if self.auto_save and self.mask_manager.masks:
            self.auto_save_masks()
            
        # Ask to save project
        if self.project_dir and self.mask_manager.masks:
            reply = QMessageBox.question(self, "Save Project?",
                "Do you want to save the project before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
                
            if reply == QMessageBox.Yes:
                self.save_project()
            elif reply == QMessageBox.Cancel:
                event.ignore()
                return
                
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application info
    app.setApplicationName("SAM2 Segmentation Tool")
    app.setOrganizationName("SAM2")
    
    window = SAM2SegmentationApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()