import numpy as np
import pandas as pd
from scipy.optimize import minimize
import time
from scipy.interpolate import CubicSpline

# ==========================================
# 1. PARAMETRELER VE KITA BİLGİLERİ
# ==========================================
N_DISTRICTS = 39
N_STATIONS = 9
N_ROUTES = N_DISTRICTS * N_STATIONS 

ALPHA = 0.5  # Trafik hassasiyeti
MU = 100.0   # Kapasite aşım cezası
CROSSING_PENALTY = 10.0 # Boğaz geçişi maliyet çarpanı (Avrupa <-> Asya)

# Sabit İlçe Sırası (data_prep ile aynı sırada olmak zorunda)
districts_list = [
    "Adalar", "Arnavutköy", "Ataşehir", "Avcılar", "Bağcılar", "Bahçelievler",
    "Bakırköy", "Başakşehir", "Bayrampaşa", "Beşiktaş", "Beykoz", "Beylikdüzü",
    "Beyoğlu", "Büyükçekmece", "Çatalca", "Çekmeköy", "Esenler", "Esenyurt",
    "Eyüpsultan", "Fatih", "Gaziosmanpaşa", "Güngören", "Kadıköy", "Kağıthane",
    "Kartal", "Küçükçekmece", "Maltepe", "Pendik", "Sancaktepe", "Sarıyer",
    "Silivri", "Sultanbeyli", "Sultangazi", "Şile", "Şişli", "Tuzla",
    "Ümraniye", "Üsküdar", "Zeytinburnu"
]

# Anadolu Yakası İlçeleri
asian_districts = {
    "Adalar", "Ataşehir", "Beykoz", "Çekmeköy", "Kadıköy", "Kartal", "Maltepe", 
    "Pendik", "Sancaktepe", "Sultanbeyli", "Şile", "Tuzla", "Ümraniye", "Üsküdar"
}

# 9 İstasyonun isimleri (Kıta tespiti için)
station_names = ["Yenibosna", "Başakşehir", "Silivri", "Hasdal", "Küçükbakkalköy", "Hekimbaşı", "Aydınlı", "Şile", "Baruthane"]
# Hangi istasyonlar Asya'da?
asian_stations = {"Küçükbakkalköy", "Hekimbaşı", "Aydınlı", "Şile"}

# Gerçek İstasyon Kapasiteleri (Proposal Q_total oranlaması)
# Toplam 12.097 tonluk kapasitenin alanlarına (A_j) göre dağıtılmış gerçekçi hali
Q_j = np.array([1800, 2200, 800, 2000, 1400, 1500, 1100, 400, 1200])

# ==========================================
# 2. VERİ YÜKLEME VE MATRİS DÜZENLEME
# ==========================================
print("📁 Mesafe matrisi ve İBB gerçek çöp verileri yükleniyor...")

try:
    D = np.load("../data/distance_matrix.npy")
except FileNotFoundError:
    print("HATA: distance_matrix.npy bulunamadı! Önce data_prep.py çalıştırın.")
    exit()

# Gerçek Çöp Miktarlarını Excel'den Okuma (d_i)
d_i = np.zeros(N_DISTRICTS)
try:
    # Veri setinde 'İlçe' ve 'Tonaj' gibi sütunlar olduğunu varsayıyoruz.
    df_ilceler = pd.read_excel("../data/ilceler.xlsx")
    col_ilce = 'Ilce_Adi' if 'Ilce_Adi' in df_ilceler.columns else df_ilceler.columns[0]
    col_tonaj = 'Tonaj' if 'Tonaj' in df_ilceler.columns else df_ilceler.columns[1] # Miktarın olduğu sütun
    
    # Excel'den gelen veriyi bizim 39'luk sıraya göre eşleştiriyoruz
    for idx, d_name in enumerate(districts_list):
        # Excel'deki ilçe adını bul, Eyüp/Eyüpsultan karmaşasını çöz
        match = df_ilceler[df_ilceler[col_ilce].astype(str).str.contains(d_name.replace("sultan", ""), case=False, na=False)]
        if not match.empty:
            d_i[idx] = float(str(match[col_tonaj].values[0]).replace(',', '.'))
        else:
            d_i[idx] = 310.0 * 365 # Bulunamazsa ortalama bir değer ata
        
    d_i = d_i / 365.0

    print(f"✅ İBB Gerçek Çöp Tonajları Yüklendi! Toplam Çöp: {np.sum(d_i):.1f} Ton")
except Exception as e:
    print("⚠️ ilceler.xlsx okunamadı veya Tonaj sütunu bulunamadı. Lütfen Excel'de miktarın olduğu sütun adını 2. sütuna veya 'Tonaj' adına getirin.")
    print("Hata detayı:", e)
    # Kod çökmesin diye fallback (Yine de random değil, sabit orantılı dağılım)
    d_i = np.linspace(200, 400, N_DISTRICTS)

# Boğaz Geçiş (Kıta) Cezasının Mesafe Matrisine Uygulanması
D_penalized = D.copy()
for i, d_name in enumerate(districts_list):
    is_district_asian = d_name in asian_districts
    for j, s_name in enumerate(station_names):
        is_station_asian = s_name in asian_stations
        
        # Eğer kıtalar uyuşmuyorsa, o yola büyük bir ceza kes (Mesafe * Çarpan)
        if is_district_asian != is_station_asian:
            D_penalized[i, j] = D[i, j] * CROSSING_PENALTY

# ==========================================
# 3. MATEMATİKSEL FONKSİYONLAR (Proposal'a Uygun)
# ==========================================

# İBB Saatlik Trafik Hızları v(t) - [6:00 ile 22:00 arası ortalama hızlar (km/h)]
# Proposal'da v_min = 27.0, v_max = 36.0 verilmiş.
v_t_dict = {
    6: 36.0, 7: 30.0, 8: 27.0, 9: 28.5, 10: 32.0, 11: 33.5, 12: 34.0, 
    13: 34.0, 14: 33.0, 15: 31.0, 16: 29.0, 17: 27.5, 18: 27.0, 19: 28.0, 
    20: 31.0, 21: 34.0, 22: 36.0
}
V_MIN = 27.0
V_MAX = 36.0

# Keskin köşeleri yok eden Pürüzsüz Spline Eğrisi oluşturuyoruz
times = list(v_t_dict.keys())
speeds = list(v_t_dict.values())
smooth_traffic_curve = CubicSpline(times, speeds)

def traffic_index(t_matrix):
    """Pürüzsüz türevlenebilir trafik yoğunluğu fonksiyonu"""
    v_t_values = smooth_traffic_curve(t_matrix)
    tau = 1.0 - ((v_t_values - V_MIN) / (V_MAX - V_MIN))
    # np.clip türevi sıfırladığı için kaldırdık, algoritmamız artık eğride serbestçe kayabilecek
    return tau

def objective_function(z):
    x = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    t = z[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS)
    
    # 1. Maliyet (D_penalized kullanarak boğaz geçişini engelliyoruz)
    tau_t = traffic_index(t)
    transport_cost = np.sum(D_penalized * (1 + ALPHA * tau_t**2) * x)
    
    # 2. Kapasite Cezası
    incoming_waste = np.sum(x, axis=0)
    excess = np.maximum(0, incoming_waste - Q_j)
    capacity_penalty = MU * np.sum(excess**2)
    
    total_cost = transport_cost + capacity_penalty
    return total_cost / 10000.0

def equality_constraint(z):
    x = z[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS)
    return np.sum(x, axis=1) - d_i

# ==========================================
# 4. ÇÖZÜCÜYÜ ÇALIŞTIRMA
# ==========================================

if __name__ == "__main__":
    print("⚙️ Optimizasyon Modeli Kuruluyor (702 Değişken)...")
    
    x0_matrix = np.zeros((N_DISTRICTS, N_STATIONS))
    for i in range(N_DISTRICTS):
        # Her ilçe için D_penalized (Boğaz cezası dahil) matrisindeki en kısa rotayı bul
        closest_station_idx = np.argmin(D_penalized[i, :]) 
        # İlçenin tüm çöpünü sadece o istasyona gönder
        x0_matrix[i, closest_station_idx] = d_i[i]
    x0 = x0_matrix.flatten()
    
    t0 = np.full(N_ROUTES, 14.0) 
    z0 = np.concatenate([x0, t0])
    
    bounds_x = [(0, None) for _ in range(N_ROUTES)]
    bounds_t = [(6.0, 22.0) for _ in range(N_ROUTES)]
    bounds = bounds_x + bounds_t
    
    constraints = {'type': 'eq', 'fun': equality_constraint}
    
    print("🚀 Çözücü başlatıldı (SLSQP)... Lütfen bekleyin...")
    start_time = time.time()
    
    result = minimize(
        objective_function, z0, method='SLSQP', bounds=bounds, constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-4, 'disp': True, 'eps': 1e-3}
    )
    
    print(f"\n⏱️ Çözüm Süresi: {time.time() - start_time:.2f} saniye")
    
    if result.success:
        print("✅ OPTİMİZASYON BAŞARILI!")
        print(f"📉 Minimum Maliyet: {result.fun:,.2f}")
        np.save("../data/optimized_x.npy", result.x[:N_ROUTES].reshape(N_DISTRICTS, N_STATIONS))
        np.save("../data/optimized_t.npy", result.x[N_ROUTES:].reshape(N_DISTRICTS, N_STATIONS))
        print("💾 Optimal dağılım 'data/' klasörüne kaydedildi!")
    else:
        print("❌ Optimizasyon Başarısız:", result.message)