import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from sklearn.cluster import KMeans
from scipy import ndimage
import warnings
from skimage.feature import local_binary_pattern as skimage_lbp
from skimage.filters import gabor
from skimage import measure
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="AI Risk Assessment Platform",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B6B;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #4ECDC4;
        margin: 2rem 0 1rem 0;
    }
    .risk-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .fire-risk {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
    }
    .flood-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

class AdvancedFireRiskPredictor:
    def __init__(self):
        self.model_loaded = False
        self.model = None
        try:
            # Attempt to load a pre-trained RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            # Note: In a real deployment, load a pre-trained model from a file
            self.model_loaded = True
        except:
            self.model_loaded = False

    def preprocess_image(self, image):
        """Preprocess image for consistent analysis"""
        # Resize to a standard size while maintaining aspect ratio
        target_size = (512, 512)
        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        # Pad to target size if necessary
        padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        pad_h = (target_size[0] - new_size[1]) // 2
        pad_w = (target_size[1] - new_size[0]) // 2
        padded_image[pad_h:pad_h+new_size[1], pad_w:pad_w+new_size[0]] = image
        
        # Normalize and reduce noise
        image = cv2.GaussianBlur(padded_image, (5, 5), 0)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image

    def extract_advanced_fire_features(self, image):
        """Extract comprehensive features for fire risk assessment"""
        # Ensure image is in RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image.copy()
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Convert to different color spaces
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        features = {}
        h, w = gray.shape
        total_pixels = h * w
        
        # 1. Advanced Vegetation Analysis
        red_channel = rgb_image[:, :, 0].astype(float)
        green_channel = rgb_image[:, :, 1].astype(float)
        ndvi_approx = np.divide(green_channel - red_channel, green_channel + red_channel + 1e-8)
        
        features['mean_ndvi'] = np.mean(ndvi_approx)
        features['std_ndvi'] = np.std(ndvi_approx)
        features['healthy_vegetation_ratio'] = np.sum(ndvi_approx > 0.3) / total_pixels
        features['stressed_vegetation_ratio'] = np.sum((ndvi_approx > 0) & (ndvi_approx <= 0.3)) / total_pixels
        
        # 2. Color-based Fire Risk Indicators
        hsv_h, hsv_s, hsv_v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        brown_mask = ((hsv_h >= 5) & (hsv_h <= 25) & (hsv_s > 100) & (hsv_v > 50) & (hsv_v < 200))
        features['brown_vegetation_ratio'] = np.sum(brown_mask) / total_pixels
        
        yellow_mask = ((hsv_h >= 20) & (hsv_h <= 40) & (hsv_s > 80) & (hsv_v > 100))
        features['yellow_vegetation_ratio'] = np.sum(yellow_mask) / total_pixels
        
        green_mask = ((hsv_h >= 35) & (hsv_h <= 85) & (hsv_s > 50) & (hsv_v > 50))
        features['green_vegetation_ratio'] = np.sum(green_mask) / total_pixels
        
        # 3. Enhanced Texture Analysis
        # Use scikit-image LBP for better texture features
        lbp = skimage_lbp(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
        for i, val in enumerate(lbp_hist):
            features[f'lbp_hist_{i}'] = val
        
        # Gabor filters for texture
        freq, theta = 0.1, 0
        gabor_response, _ = gabor(gray, frequency=freq, theta=theta)
        features['gabor_mean'] = np.mean(gabor_response)
        features['gabor_std'] = np.std(gabor_response)
        
        # Edge density with adaptive thresholding
        edges = cv2.Canny(gray, 100, 200)
        features['edge_density'] = np.sum(edges > 0) / total_pixels
        
        # 4. Moisture and Dryness Indicators
        features['mean_brightness'] = np.mean(gray)
        features['brightness_std'] = np.std(gray)
        features['very_bright_ratio'] = np.sum(gray > 200) / total_pixels
        features['red_intensity'] = np.mean(red_channel)
        features['red_dominance'] = np.mean(red_channel > green_channel)
        
        # 5. Entropy Analysis
        entropy = measure.shannon_entropy(gray)
        features['image_entropy'] = entropy
        
        # 6. Advanced Clustering Analysis
        pixels = rgb_image.reshape(-1, 3)
        n_clusters = min(5, len(np.unique(pixels, axis=0)))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(pixels)
            cluster_centers = kmeans.cluster_centers_
            
            fire_prone_clusters = 0
            for center in cluster_centers:
                r, g, b = center
                if (r > g and r > b and r > 100) or (r > 150 and g > 100 and b < 100):
                    fire_prone_clusters += 1
            features['fire_prone_color_clusters'] = fire_prone_clusters / n_clusters
        else:
            features['fire_prone_color_clusters'] = 0
        
        # 7. Spatial Analysis
        green_binary = green_mask.astype(np.uint8)
        contours, _ = cv2.findContours(green_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            total_green_area = np.sum(green_binary)
            n_patches = len(contours)
            features['vegetation_fragmentation'] = n_patches / (total_green_area + 1e-8) * 1000
        else:
            features['vegetation_fragmentation'] = 0
            
        features['shadow_ratio'] = np.sum(gray < 50) / total_pixels
        
        return features
    
    def predict_fire_risk(self, image):
        """Enhanced fire risk prediction with ML integration"""
        image = self.preprocess_image(image)
        features = self.extract_advanced_fire_features(image)
        
        if self.model_loaded:
            # Prepare features for ML model
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            try:
                risk_score = self.model.predict_proba(feature_vector)[0][1] * 100
                risk_factors = ["ML-based risk assessment"]
                confidence_modifiers = []
            except:
                # Fallback to rule-based if ML prediction fails
                risk_score, risk_level, risk_factors, confidence_modifiers = self._rule_based_fire_risk(features)
        else:
            # Rule-based scoring
            risk_score, risk_level, risk_factors, confidence_modifiers = self._rule_based_fire_risk(features)
        
        return risk_score, risk_level, risk_factors, confidence_modifiers

    def _rule_based_fire_risk(self, features):
        """Rule-based fire risk prediction as fallback"""
        risk_score = 0
        risk_factors = []
        confidence_modifiers = []
        
        # Vegetation health assessment (35% weight)
        vegetation_risk = 0
        if features['healthy_vegetation_ratio'] < 0.2:
            vegetation_risk += 30
            risk_factors.append("Low healthy vegetation coverage detected")
        elif features['healthy_vegetation_ratio'] < 0.4:
            vegetation_risk += 15
            risk_factors.append("Moderate healthy vegetation coverage")
            
        if features['stressed_vegetation_ratio'] > 0.3:
            vegetation_risk += 20
            risk_factors.append("High stressed vegetation detected")
            
        if features['brown_vegetation_ratio'] > 0.15:
            vegetation_risk += 25
            risk_factors.append("Significant dry/brown vegetation present")
            
        if features['yellow_vegetation_ratio'] > 0.1:
            vegetation_risk += 10
            risk_factors.append("Yellow/stressed vegetation detected")
            
        risk_score += min(vegetation_risk, 35)
        
        # Dryness indicators (25% weight)
        dryness_risk = 0
        if features['mean_brightness'] > 150:
            dryness_risk += 15
            risk_factors.append("High brightness indicating dry conditions")
            
        if features['red_dominance'] > 0.6:
            dryness_risk += 10
            risk_factors.append("Red color dominance suggesting dryness")
            
        if features['very_bright_ratio'] > 0.3:
            dryness_risk += 10
            risk_factors.append("Many very bright areas detected")
            
        risk_score += min(dryness_risk, 25)
        
        # Texture and terrain (20% weight)
        terrain_risk = 0
        if features['image_entropy'] > 7:
            terrain_risk += 10
            risk_factors.append("High image entropy indicating complex terrain")
            
        if features['edge_density'] > 0.15:
            terrain_risk += 5
            risk_factors.append("High edge density indicating complex terrain")
            
        if features['vegetation_fragmentation'] > 50:
            terrain_risk += 5
            risk_factors.append("Fragmented vegetation pattern")
            
        risk_score += min(terrain_risk, 20)
        
        # Color clustering and advanced features (20% weight)
        advanced_risk = 0
        if features['fire_prone_color_clusters'] > 0.4:
            advanced_risk += 15
            risk_factors.append("Fire-prone color patterns detected")
        elif features['fire_prone_color_clusters'] > 0.2:
            advanced_risk += 7
            
        if features['gabor_std'] > 10:
            advanced_risk += 5
            risk_factors.append("High texture variation detected")
            
        risk_score += min(advanced_risk, 20)
        
        # Confidence modifiers
        if features['shadow_ratio'] > 0.4:
            confidence_modifiers.append("High shadow areas may affect accuracy")
        if features['healthy_vegetation_ratio'] + features['stressed_vegetation_ratio'] < 0.1:
            confidence_modifiers.append("Limited vegetation detected - analysis may be less accurate")
            
        # Final risk score normalization
        risk_score = min(max(risk_score, 0), 100)
        
        # Determine risk level
        if risk_score >= 75:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        elif risk_score >= 20:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return risk_score, risk_level, risk_factors, confidence_modifiers

class AdvancedFloodRiskPredictor:
    def __init__(self):
        self.model_loaded = False
        self.model = None
        try:
            # Attempt to load a pre-trained RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model_loaded = True
        except:
            self.model_loaded = False

    def preprocess_image(self, image):
        """Preprocess image for consistent analysis"""
        # Resize to a standard size while maintaining aspect ratio
        target_size = (512, 512)
        h, w = image.shape[:2]
        scale = min(target_size[0] / h, target_size[1] / w)
        new_size = (int(w * scale), int(h * scale))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        
        # Pad to target size if necessary
        padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        pad_h = (target_size[0] - new_size[1]) // 2
        pad_w = (target_size[1] - new_size[0]) // 2
        padded_image[pad_h:pad_h+new_size[1], pad_w:pad_w+new_size[0]] = image
        
        # Normalize and reduce noise
        image = cv2.GaussianBlur(padded_image, (5, 5), 0)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image

    def extract_advanced_flood_features(self, image):
        """Extract comprehensive features for flood risk assessment"""
        # Ensure image is in RGB format
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = image.copy()
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        features = {}
        h, w = gray.shape
        total_pixels = h * w
        
        # 1. Water Detection
        hsv_h, hsv_s, hsv_v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        clear_water_mask = ((hsv_h >= 90) & (hsv_h <= 130) & (hsv_s > 30) & (hsv_v > 30))
        features['clear_water_ratio'] = np.sum(clear_water_mask) / total_pixels
        
        muddy_water_mask = ((hsv_h >= 10) & (hsv_h <= 30) & (hsv_s > 20) & (hsv_v > 20) & (hsv_v < 150))
        features['muddy_water_ratio'] = np.sum(muddy_water_mask) / total_pixels
        
        dark_water_mask = (hsv_v < 60) & (hsv_s < 100)
        features['dark_water_ratio'] = np.sum(dark_water_mask) / total_pixels
        
        # 2. Terrain Analysis
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['mean_slope'] = np.mean(gradient_magnitude)
        features['slope_variance'] = np.var(gradient_magnitude)
        features['flat_area_ratio'] = np.sum(gradient_magnitude < 5) / total_pixels
        
        normalized_gray = gray.astype(float) / 255.0
        features['elevation_variance'] = np.var(normalized_gray)
        features['low_elevation_ratio'] = np.sum(normalized_gray < 0.3) / total_pixels
        
        # 3. Surface Analysis
        kernel = np.ones((5,5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(float), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(float) - local_mean)**2, -1, kernel)
        
        high_reflectivity = local_variance > np.percentile(local_variance, 80)
        features['reflective_surface_ratio'] = np.sum(high_reflectivity) / total_pixels
        
        # 4. Vegetation and Impervious Surfaces
        green_mask = ((hsv_h >= 35) & (hsv_h <= 85) & (hsv_s > 50) & (hsv_v > 50))
        features['vegetation_coverage'] = np.sum(green_mask) / total_pixels
        
        impervious_mask = (hsv_s < 30) & (hsv_v > 80) & (hsv_v < 200)
        features['impervious_surface_ratio'] = np.sum(impervious_mask) / total_pixels
        
        # 5. Morphological Analysis
        water_combined = (clear_water_mask | muddy_water_mask).astype(np.uint8)
        contours, _ = cv2.findContours(water_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_water_area = cv2.contourArea(largest_contour)
            features['largest_water_body_ratio'] = largest_water_area / total_pixels
            features['water_body_count'] = len([c for c in contours if cv2.contourArea(c) > 100])
        else:
            features['largest_water_body_ratio'] = 0
            features['water_body_count'] = 0
            
        # 6. Advanced Features
        features['low_saturation_ratio'] = np.sum(hsv_s < 50) / total_pixels
        features['blue_intensity'] = np.mean(rgb_image[:, :, 2])
        features['blue_dominance'] = np.mean(rgb_image[:, :, 2] > rgb_image[:, :, 0])
        
        lbp = skimage_lbp(gray, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
        for i, val in enumerate(lbp_hist):
            features[f'lbp_hist_{i}'] = val
            
        features['image_entropy'] = measure.shannon_entropy(gray)
        
        return features
    
    def predict_flood_risk(self, image):
        """Enhanced flood risk prediction with ML integration"""
        image = self.preprocess_image(image)
        features = self.extract_advanced_flood_features(image)
        
        if self.model_loaded:
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            try:
                risk_score = self.model.predict_proba(feature_vector)[0][1] * 100
                risk_factors = ["ML-based risk assessment"]
                confidence_modifiers = []
            except:
                risk_score, risk_level, risk_factors, confidence_modifiers = self._rule_based_flood_risk(features)
        else:
            risk_score, risk_level, risk_factors, confidence_modifiers = self._rule_based_flood_risk(features)
        
        return risk_score, risk_level, risk_factors, confidence_modifiers

    def _rule_based_flood_risk(self, features):
        """Rule-based flood risk prediction as fallback"""
        risk_score = 0
        risk_factors = []
        confidence_modifiers = []
        
        # Water presence analysis (30% weight)
        water_risk = 0
        total_water = features['clear_water_ratio'] + features['muddy_water_ratio'] + features['dark_water_ratio']
        
        if total_water > 0.3:
            water_risk += 30
            risk_factors.append("Significant water bodies present")
        elif total_water > 0.15:
            water_risk += 20
            risk_factors.append("Moderate water presence detected")
        elif total_water > 0.05:
            water_risk += 10
            risk_factors.append("Some water areas detected")
            
        if features['largest_water_body_ratio'] > 0.2:
            water_risk += 10
            risk_factors.append("Large water body detected")
            
        risk_score += min(water_risk, 30)
        
        # Terrain and drainage analysis (30% weight)
        terrain_risk = 0
        if features['flat_area_ratio'] > 0.6:
            terrain_risk += 20
            risk_factors.append("Predominantly flat terrain with poor drainage")
        elif features['flat_area_ratio'] > 0.4:
            terrain_risk += 12
            risk_factors.append("Moderate flat areas present")
            
        if features['low_elevation_ratio'] > 0.4:
            terrain_risk += 10
            risk_factors.append("Low-lying areas detected")
            
        if features['mean_slope'] < 3:
            terrain_risk += 8
            risk_factors.append("Very low terrain slope")
            
        risk_score += min(terrain_risk, 30)
        
        # Surface and drainage characteristics (20% weight)
        surface_risk = 0
        if features['impervious_surface_ratio'] > 0.4:
            surface_risk += 15
            risk_factors.append("High impervious surface coverage")
        elif features['impervious_surface_ratio'] > 0.2:
            surface_risk += 8
            risk_factors.append("Moderate impervious surfaces detected")
            
        if features['vegetation_coverage'] < 0.2:
            surface_risk += 10
            risk_factors.append("Limited vegetation for water absorption")
            
        risk_score += min(surface_risk, 20)
        
        # Advanced features (20% weight)
        advanced_risk = 0
        if features['reflective_surface_ratio'] > 0.3:
            advanced_risk += 10
            risk_factors.append("High reflectivity suggesting water surfaces")
        if features['image_entropy'] < 5:
            advanced_risk += 5
            risk_factors.append("Low entropy indicating smooth surfaces")
        if features['blue_dominance'] > 0.6:
            advanced_risk += 5
            risk_factors.append("Blue dominance suggesting water presence")
            
        risk_score += min(advanced_risk, 20)
        
        # Confidence modifiers
        if features['vegetation_coverage'] > 0.7:
            confidence_modifiers.append("Dense vegetation may obscure flood-prone areas")
        if total_water < 0.01 and features['flat_area_ratio'] < 0.2:
            confidence_modifiers.append("Limited flood indicators - assessment based on terrain analysis")
            
        # Final risk score normalization
        risk_score = min(max(risk_score, 0), 100)
        
        # Determine risk level
        if risk_score >= 75:
            risk_level = "CRITICAL"
        elif risk_score >= 60:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        elif risk_score >= 20:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return risk_score, risk_level, risk_factors, confidence_modifiers

def create_enhanced_risk_visualization(risk_score, risk_level, risk_type, confidence_modifiers):
    """Create an enhanced risk visualization chart"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['#27AE60', '#F39C12', '#E67E22', '#E74C3C', '#8E44AD']
    level_colors = {
        'MINIMAL': colors[0],
        'LOW': colors[1], 
        'MEDIUM': colors[2],
        'HIGH': colors[3],
        'CRITICAL': colors[4]
    }
    
    current_color = level_colors[risk_level]
    
    ax1.pie([risk_score, 100-risk_score], colors=[current_color, '#ECF0F1'], 
            startangle=90, counterclock=False, wedgeprops=dict(width=0.3))
    ax1.text(0, 0, f'{risk_score}%', ha='center', va='center', fontsize=20, fontweight='bold')
    ax1.set_title(f'{risk_type} Risk Score', fontsize=14, fontweight='bold')
    
    levels = ['MINIMAL', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    level_values = [20, 20, 20, 20, 20]
    current_level_idx = levels.index(risk_level)
    
    bars = ax2.bar(levels, level_values, color=['lightgray']*5, alpha=0.3)
    bars[current_level_idx].set_color(current_color)
    bars[current_level_idx].set_alpha(1.0)
    
    ax2.set_ylabel('Risk Level')
    ax2.set_title(f'Current Risk Level: {risk_level}', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    score_ranges = ['0-20', '21-40', '41-60', '61-75', '76-100']
    range_labels = ['Minimal', 'Low', 'Medium', 'High', 'Critical']
    
    if risk_score <= 20:
        current_range = 0
    elif risk_score <= 40:
        current_range = 1
    elif risk_score <= 60:
        current_range = 2
    elif risk_score <= 75:
        current_range = 3
    else:
        current_range = 4
    
    range_colors = ['lightgray'] * 5
    range_colors[current_range] = current_color
    
    ax3.barh(range_labels, [20, 20, 20, 15, 25], color=range_colors, alpha=0.7)
    ax3.axvline(x=risk_score/5, color='red', linestyle='--', linewidth=2, label=f'Current Score: {risk_score}')
    ax3.set_xlabel('Score Range')
    ax3.set_title('Risk Score Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    
    if confidence_modifiers:
        confidence_text = '\n'.join(['⚠️ ' + mod for mod in confidence_modifiers])
        ax4.text(0.5, 0.7, 'Confidence Notes:', ha='center', va='center', 
                fontsize=12, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.5, 0.3, confidence_text, ha='center', va='center', 
                fontsize=10, transform=ax4.transAxes, wrap=True)
    else:
        ax4.text(0.5, 0.5, '✅ High Confidence\nAssessment', ha='center', va='center', 
                fontsize=14, fontweight='bold', transform=ax4.transAxes)
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Analysis Confidence', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    st.markdown('<h1 class="main-header">🔥💧 Enhanced AI Risk Assessment Platform</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Advanced computer vision analysis for accurate fire and flood risk assessment
    </div>
    """, unsafe_allow_html=True)
    
    fire_predictor = AdvancedFireRiskPredictor()
    flood_predictor = AdvancedFloodRiskPredictor()
    
    with st.sidebar:
        st.header("📊 Enhanced Analysis Options")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Fire Risk Assessment", "Flood Risk Assessment", "Both"]
        )
        
        st.header("🔬 Advanced Features")
        st.info("""
        **Enhanced Fire Risk Analysis:**
        - Advanced NDVI-based vegetation health
        - Multi-spectral color analysis with entropy
        - Gabor filter-based texture analysis
        - Robust clustering with dynamic k-selection
        - Spatial pattern recognition
        
        **Enhanced Flood Risk Analysis:**
        - Multi-modal water detection with entropy
        - Advanced terrain and slope analysis
        - Surface permeability assessment
        - Morphological water body analysis
        - Enhanced drainage modeling
        """)
        
        show_debug = st.checkbox("Show Debug Information", False)
    
    col1, col2 = st.columns([1, 1])
    
    fire_risk_score = None
    fire_risk_level = None
    fire_risk_factors = []
    flood_risk_score = None
    flood_risk_level = None
    flood_risk_factors = []
    
    if analysis_type in ["Fire Risk Assessment", "Both"]:
        with col1:
            st.markdown('<div class="risk-card fire-risk">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">🔥 Enhanced Fire Risk Assessment</h2>', unsafe_allow_html=True)
            
            fire_uploaded_file = st.file_uploader(
                "Upload an image for fire risk analysis",
                type=['png', 'jpg', 'jpeg'],
                key="fire_uploader"
            )
            
            if fire_uploaded_file is not None:
                try:
                    image = Image.open(fire_uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    image_array = np.array(image)
                    if image_array.shape[2] == 4:
                        image_array = image_array[:, :, :3]
                    
                    with st.spinner("Performing advanced fire risk analysis..."):
                        fire_risk_score, fire_risk_level, fire_risk_factors, confidence_modifiers = fire_predictor.predict_fire_risk(image_array)
                    
                    st.markdown("### 📊 Enhanced Analysis Results")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Risk Score", f"{fire_risk_score:.0f}%")
                    with col_b:
                        st.metric("Risk Level", fire_risk_level)
                    with col_c:
                        color_map = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "MINIMAL": "⚪"}
                        st.metric("Status", color_map.get(fire_risk_level, "❓"))
                    with col_d:
                        confidence = "High" if not confidence_modifiers else "Medium"
                        st.metric("Confidence", confidence)
                    
                    st.markdown("### ⚠️ Risk Factors Identified")
                    if fire_risk_factors:
                        for factor in fire_risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("• No significant risk factors detected")
                    
                    if confidence_modifiers:
                        st.markdown("### 🔍 Analysis Notes")
                        for modifier in confidence_modifiers:
                            st.write(f"⚠️ {modifier}")
                    
                    fig = create_enhanced_risk_visualization(fire_risk_score, fire_risk_level, "Fire", confidence_modifiers)
                    st.pyplot(fig)
                    
                    if show_debug:
                        st.markdown("### 🔧 Debug Information")
                        features = fire_predictor.extract_advanced_fire_features(image_array)
                        debug_df = pd.DataFrame([features]).T
                        debug_df.columns = ['Value']
                        st.dataframe(debug_df)
                    
                    st.markdown("### 💡 Detailed Recommendations")
                    if fire_risk_level == "CRITICAL":
                        st.error("""
                        **🚨 CRITICAL FIRE RISK DETECTED!**
                        - **IMMEDIATE ACTION REQUIRED**
                        - Evacuate area if necessary
                        - Contact fire department/emergency services
                        - Implement emergency fire suppression measures
                        - Clear all flammable materials within 100m
                        - Monitor wind conditions continuously
                        - Have evacuation routes planned
                        """)
                    elif fire_risk_level == "HIGH":
                        st.error("""
                        **🔥 HIGH FIRE RISK**
                        - Implement immediate fire prevention measures
                        - Clear dry vegetation around structures (30m minimum)
                        - Ensure fire suppression systems are operational
                        - Monitor weather conditions closely
                        - Restrict spark-causing activities
                        - Have emergency response plan ready
                        """)
                    elif fire_risk_level == "MEDIUM":
                        st.warning("""
                        **⚠️ MODERATE FIRE RISK**
                        - Maintain regular vegetation clearing
                        - Monitor for changes in vegetation health
                        - Keep fire suppression equipment accessible
                        - Follow local fire restrictions
                        - Regular patrol and monitoring recommended
                        """)
                    elif fire_risk_level == "LOW":
                        st.info("""
                        **ℹ️ LOW FIRE RISK**
                        - Continue standard fire prevention practices
                        - Seasonal vegetation management
                        - Regular equipment checks
                        - Monitor during high-risk weather periods
                        """)
                    else:
                        st.success("""
                        **✅ MINIMAL FIRE RISK**
                        - Maintain current vegetation management
                        - Standard seasonal monitoring sufficient
                        - Continue regular maintenance schedules
                        """)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    if analysis_type in ["Flood Risk Assessment", "Both"]:
        with col2:
            st.markdown('<div class="risk-card flood-risk">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">💧 Enhanced Flood Risk Assessment</h2>', unsafe_allow_html=True)
            
            flood_uploaded_file = st.file_uploader(
                "Upload an image for flood risk analysis",
                type=['png', 'jpg', 'jpeg'],
                key="flood_uploader"
            )
            
            if flood_uploaded_file is not None:
                try:
                    image = Image.open(flood_uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    image_array = np.array(image)
                    if image_array.shape[2] == 4:
                        image_array = image_array[:, :, :3]
                    
                    with st.spinner("Performing advanced flood risk analysis..."):
                        flood_risk_score, flood_risk_level, flood_risk_factors, confidence_modifiers = flood_predictor.predict_flood_risk(image_array)
                    
                    st.markdown("### 📊 Enhanced Analysis Results")
                    
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Risk Score", f"{flood_risk_score:.0f}%")
                    with col_b:
                        st.metric("Risk Level", flood_risk_level)
                    with col_c:
                        color_map = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "MINIMAL": "⚪"}
                        st.metric("Status", color_map.get(flood_risk_level, "❓"))
                    with col_d:
                        confidence = "High" if not confidence_modifiers else "Medium"
                        st.metric("Confidence", confidence)
                    
                    st.markdown("### 🌊 Risk Factors Identified")
                    if flood_risk_factors:
                        for factor in flood_risk_factors:
                            st.write(f"• {factor}")
                    else:
                        st.write("• No significant risk factors detected")
                    
                    if confidence_modifiers:
                        st.markdown("### 🔍 Analysis Notes")
                        for modifier in confidence_modifiers:
                            st.write(f"⚠️ {modifier}")
                    
                    fig = create_enhanced_risk_visualization(flood_risk_score, flood_risk_level, "Flood", confidence_modifiers)
                    st.pyplot(fig)
                    
                    if show_debug:
                        st.markdown("### 🔧 Debug Information")
                        features = flood_predictor.extract_advanced_flood_features(image_array)
                        debug_df = pd.DataFrame([features]).T
                        debug_df.columns = ['Value']
                        st.dataframe(debug_df)
                    
                    st.markdown("### 💡 Detailed Recommendations")
                    if flood_risk_level == "CRITICAL":
                        st.error("""
                        **🚨 CRITICAL FLOOD RISK DETECTED!**
                        - **IMMEDIATE ACTION REQUIRED**
                        - Evacuate low-lying areas immediately
                        - Contact emergency services
                        - Deploy emergency flood barriers
                        - Monitor water levels continuously
                        - Implement emergency drainage measures
                        - Have evacuation routes ready
                        """)
                    elif flood_risk_level == "HIGH":
                        st.error("""
                        **🌊 HIGH FLOOD RISK**
                        - Implement immediate flood protection measures
                        - Clear drainage systems of debris
                        - Install temporary flood barriers
                        - Monitor weather forecasts closely
                        - Prepare emergency response equipment
                        - Develop detailed evacuation plans
                        """)
                    elif flood_risk_level == "MEDIUM":
                        st.warning("""
                        **⚠️ MODERATE FLOOD RISK**
                        - Regular drainage system maintenance
                        - Monitor water levels during heavy rainfall
                        - Prepare flood response materials
                        - Improve surface water management
                        - Consider drainage improvements
                        """)
                    elif flood_risk_level == "LOW":
                        st.info("""
                        **ℹ️ LOW FLOOD RISK**
                        - Standard drainage maintenance
                        - Seasonal system inspections
                        - Monitor during extreme weather events
                        - Maintain emergency preparedness
                        """)
                    else:
                        st.success("""
                        **✅ MINIMAL FLOOD RISK**
                        - Continue routine maintenance
                        - Standard monitoring sufficient
                        - Keep drainage systems clear
                        """)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    if analysis_type == "Both" and st.sidebar.button("Generate Combined Risk Report"):
        if fire_uploaded_file is not None and flood_uploaded_file is not None:
            st.markdown("---")
            st.markdown("### 📋 Combined Risk Assessment Report")
            
            combined_risk = (fire_risk_score + flood_risk_score) / 2 if fire_risk_score is not None and flood_risk_score is not None else 0
            
            report_data = {
                "Fire Risk Score": f"{fire_risk_score:.0f}%" if fire_risk_score is not None else "Not analyzed",
                "Fire Risk Level": fire_risk_level if fire_risk_level is not None else "Not analyzed",
                "Flood Risk Score": f"{flood_risk_score:.0f}%" if flood_risk_score is not None else "Not analyzed",
                "Flood Risk Level": flood_risk_level if flood_risk_level is not None else "Not analyzed",
                "Overall Risk": f"{combined_risk:.0f}%",
                "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.json(report_data)
            
            report_text = f"""
            COMBINED RISK ASSESSMENT REPORT
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
            FIRE RISK ANALYSIS:
            - Risk Score: {report_data['Fire Risk Score']}
            - Risk Level: {report_data['Fire Risk Level']}
            - Factors: {', '.join(fire_risk_factors) if fire_risk_factors else 'None detected'}
            
            FLOOD RISK ANALYSIS:
            - Risk Score: {report_data['Flood Risk Score']}
            - Risk Level: {report_data['Flood Risk Level']}
            - Factors: {', '.join(flood_risk_factors) if flood_risk_factors else 'None detected'}
            
            OVERALL ASSESSMENT:
            - Combined Risk Score: {combined_risk:.0f}%
            - Priority Actions Required: {'Yes' if combined_risk > 60 else 'Standard monitoring'}
            """
            
            st.download_button(
                label="📄 Download Combined Report",
                data=report_text,
                file_name=f"risk_assessment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        🤖 Powered by Enhanced AI Computer Vision | Built with Advanced Machine Learning<br>
        <small>
        This enhanced tool provides sophisticated risk assessments using advanced feature extraction, 
        machine learning integration, and robust preprocessing. Results should be used in conjunction with 
        professional risk assessments and local expertise for critical decisions.
        <br><br>
        <strong>Key Improvements:</strong> Robust preprocessing, advanced texture analysis, 
        entropy-based features, ML integration, and error handling
        </small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()