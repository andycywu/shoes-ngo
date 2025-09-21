"""
基本的 API 測試檔案
"""

import pytest
import requests
from pathlib import Path
import io
from PIL import Image

# 測試配置
API_BASE = "http://localhost:7860"
TEST_IMAGES_DIR = Path(__file__).parent.parent / "test_images"

def create_test_image():
    """建立測試用的圖片"""
    img = Image.new('RGB', (224, 224), color='red')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    return buf

class TestAPI:
    """API 測試類別"""
    
    def test_health_check(self):
        """測試健康檢查端點"""
        try:
            response = requests.get(f"{API_BASE}/")
            assert response.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("API 服務器未啟動")
    
    def test_analyze_endpoint(self):
        """測試分析端點"""
        try:
            # 建立測試圖片
            test_image = create_test_image()
            
            files = {
                'img': ('test.jpg', test_image, 'image/jpeg')
            }
            data = {
                'user_email': 'test@example.com',
                'brand': 'Nike',
                'model_name': 'Test Shoe'
            }
            
            response = requests.post(f"{API_BASE}/analyze", files=files, data=data)
            
            # 檢查回應
            assert response.status_code == 200
            json_data = response.json()
            
            # 檢查必要欄位
            assert 'is_sneaker' in json_data
            assert 'defects' in json_data
            assert 'vlm' in json_data
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API 服務器未啟動")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])