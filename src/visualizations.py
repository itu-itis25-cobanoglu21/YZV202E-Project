import numpy as np
import folium
import os

# Koordinat verileri (Haritada çizebilmek için)
districts_coords = {
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

station_coords = [
    (41.0024, 28.8213), (41.0822, 28.7997), (41.0740, 28.2460), 
    (41.0881, 28.9482), (40.9780, 29.1120), (41.0450, 29.1000), 
    (40.8700, 29.3100), (41.1750, 29.6100), (40.9810, 28.8820)
]
station_names = ["Yenibosna", "Başakşehir", "Silivri", "Hasdal", "Küçükbakkalköy", "Hekimbaşı", "Aydınlı", "Şile", "Baruthane"]

def create_map():
    print("🗺️ Optimizasyon sonuçları okunuyor ve harita oluşturuluyor...")
    
    try:
        x_opt = np.load("../data/optimized_x.npy")
        t_opt = np.load("../data/optimized_t.npy")
    except FileNotFoundError:
        print("HATA: optimize edilmiş dosyalar bulunamadı! Önce model.py'yi çalıştırın.")
        return

    # İstanbul merkezli bir harita oluştur
    m = folium.Map(location=[41.0082, 28.9784], zoom_start=10, tiles="CartoDB dark_matter")

    # İstasyonları haritaya kırmızı ikonlarla ekle
    for idx, coord in enumerate(station_coords):
        folium.Marker(
            location=coord,
            popup=f"İstasyon: {station_names[idx]}",
            icon=folium.Icon(color="red", icon="trash")
        ).add_to(m)

    # 39 İlçe için hesaplanan rotaları (sadece atık gidenleri) çiz
    districts_list = list(districts_coords.keys())
    
    active_routes = 0
    for i, d_name in enumerate(districts_list):
        d_coord = districts_coords[d_name]
        
        for j, s_coord in enumerate(station_coords):
            tonnage = x_opt[i, j]
            time_val = t_opt[i, j]
            
            # Eğer o istasyona 10 tondan fazla çöp gidiyorsa rotayı çiz 
            # (Çok küçük virgüllü sayıları (1e-6) haritayı kirletmesin diye yoksayıyoruz)
            if tonnage > 10.0:
                active_routes += 1
                # Çizgi kalınlığı taşınan yüke göre değişsin
                weight = max(1, tonnage / 50.0)

                # EKLENEN KISIM: 12.6 gibi ondalıklı saati Saat ve Dakikaya çeviriyoruz
                hours = int(time_val)
                minutes = int((time_val - hours) * 60)
                time_str = f"{hours:02d}:{minutes:02d}"
                
                folium.PolyLine(
                    locations=[d_coord, s_coord],
                    color="#00ff00", # Neon yeşil (geliştirdiğin oyunun neon-noir estetiğine uygun!)
                    weight=weight,
                    opacity=0.6,
                    tooltip=f"{d_name} -> {station_names[j]}<br>Tonaj: {tonnage:.1f} ton<br>Kalkış: {time_str}"
                ).add_to(m)

    # Haritayı kaydet
    os.makedirs("../data", exist_ok=True)
    map_path = "../data/istanbul_waste_map.html"
    m.save(map_path)
    
    print(f"✅ Harita başarıyla oluşturuldu! Toplam aktif rota sayısı: {active_routes}")
    print(f"📂 Lütfen tarayıcınızda şu dosyayı açın: {os.path.abspath(map_path)}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    create_map()