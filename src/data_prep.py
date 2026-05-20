import numpy as np
import pandas as pd
import requests
import time
import os

# ==========================================
# AYARLAR (SETTINGS)
USE_GEOCODING_API = False 
# ==========================================

def get_district_coordinates():
    """39 İlçenin sabit koordinat haritasını döndürür."""
    print("📍 39 İlçenin koordinatları yükleniyor...")
    return {
        "Adalar": (40.8744, 29.1320), "Arnavutköy": (41.1853, 28.7385), "Ataşehir": (40.9845, 29.1065),
        "Avcılar": (40.9796, 28.7214), "Bağcılar": (41.0340, 28.8415), "Bahçelievler": (41.0003, 28.8504),
        "Bakırköy": (40.9833, 28.8686), "Başakşehir": (41.0827, 28.7963), "Bayrampaşa": (41.0346, 28.9048),
        "Beşiktaş": (41.0435, 29.0069), "Beykoz": (41.1179, 29.0963), "Beylikdüzü": (40.9902, 28.6416),
        "Beyoğlu": (41.0337, 28.9776), "Büyükçekmece": (41.0210, 28.5786), "Çatalca": (41.1440, 28.4616),
        "Çekmeköy": (41.0351, 29.1764), "Esenler": (41.0384, 28.8824), "Esenyurt": (41.0343, 28.6801),
        "Eyüpsultan": (41.0475, 28.9329), "Fatih": (41.0156, 28.9443), "Gaziosmanpaşa": (41.0573, 28.9060),
        "Güngören": (41.0223, 28.8724), "Kadıköy": (40.9880, 29.0270), "Kağıthane": (41.0811, 28.9754),
        "Kartal": (40.8885, 29.1866), "Küçükçekmece": (41.0028, 28.7842), "Maltepe": (40.9257, 29.1362),
        "Pendik": (40.8770, 29.2346), "Sancaktepe": (40.9897, 29.2255), "Sarıyer": (41.1687, 29.0504),
        "Silivri": (41.0743, 28.2464), "Sultanbeyli": (40.9669, 29.2662), "Sultangazi": (41.1065, 28.8837),
        "Şile": (41.1754, 29.6128), "Şişli": (41.0610, 28.9878), "Tuzla": (40.8166, 29.3097),
        "Ümraniye": (41.0256, 29.0984), "Üsküdar": (41.0267, 29.0158), "Zeytinburnu": (40.9896, 28.9026)
    }

def get_station_coordinates():
    """Proposal'daki 9 Katı Atık Aktarma İstasyonunun kesin koordinatlarını döndürür."""
    print("📍 9 İstasyonun koordinatları yükleniyor...")
    return [
        (41.0024, 28.8213), # Yenibosna
        (41.0822, 28.7997), # Başakşehir
        (41.0740, 28.2460), # Silivri
        (41.0881, 28.9482), # Hasdal
        (40.9780, 29.1120), # Küçükbakkalköy
        (41.0450, 29.1000), # Hekimbaşı
        (40.8700, 29.3100), # Aydınlı
        (41.1750, 29.6100), # Şile
        (40.9810, 28.8820)  # Baruthane
    ]

def get_osrm_distance(coord1, coord2):
    """İki float koordinat tuple'ı arasındaki mesafeyi OSRM API'den çeker."""
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()["routes"][0]["distance"] / 1000.0
    except Exception:
        pass
    
    return np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2) * 111.0 

def create_distance_matrix():
    district_dict = get_district_coordinates()
    station_coords = get_station_coordinates()
    
    district_coords = list(district_dict.values())
    
    print(f"\n🚗 Toplam {len(district_coords)} ilçe x {len(station_coords)} istasyon için rotalar OSRM'den çekiliyor...")
    
    # 39x9'luk matris oluştur
    D = np.zeros((len(district_coords), len(station_coords)))
    
    for i, d_coord in enumerate(district_coords):
        for j, s_coord in enumerate(station_coords):
            D[i, j] = get_osrm_distance(d_coord, s_coord)
            time.sleep(1.0) # OSRM API rate limitine (1 req/sec) uymak için
            
        if (i+1) % 5 == 0:
            print(f"{i+1} ilçenin hesaplaması tamamlandı...")
            
    os.makedirs("../data", exist_ok=True)
    np.save("../data/distance_matrix.npy", D)
    print("\n✅ Matris 'data/distance_matrix.npy' olarak kaydedildi.")
    return D

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    D = create_distance_matrix()
    print("Mesafe Matrisi Boyutu:", D.shape) # Burası kesinlikle (39, 9) çıkmalı!