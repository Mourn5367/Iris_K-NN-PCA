import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import random

# 데이터 로드
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = ['꽃받침 길이', '꽃받침 너비', '꽃잎 길이', '꽃잎 너비']
target_names = ['세토사', '버시컬러', '버지니카']

# 한글 폰트 설정
try:
    font_path = 'Maplestory Light.ttf'
    font = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    font_name = font.get_name()
    plt.rc('font', family=font_name)
    # 마이너스 기호 깨짐 방지
    plt.rc('axes', unicode_minus=False)
except:
    print("한글 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")

# 다양한 랜덤 시드 설정
random_seeds = random.sample(range(1, 1001), 11)
k_values = range(1, 30, 2)  # 원래 코드와 동일한 k 값 범위

# 결과를 저장할 데이터프레임 생성
results = pd.DataFrame(columns=['random_seed', 'k', 'original_accuracy', 'pca1_accuracy', 'pca2_accuracy'])

# 각 랜덤 시드에 대해 KNN 모델 평가
for seed in random_seeds:
    print(f"\n랜덤 시드 {seed}에 대한 평가:")
    
    # 데이터 분할 - 학습용(80%)과 테스트용(20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed)
    
    # 데이터 표준화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA 적용
    pca = PCA()
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # 설명된 분산 비율 출력
    explained_variance = pca.explained_variance_ratio_
    print(f"PC1 + PC2의 누적 설명 분산: {explained_variance[0] + explained_variance[1]:.4f}")
    
    # k값에 따른 정확도 평가
    for k in k_values:
        # 원본 데이터
        knn_original = KNeighborsClassifier(n_neighbors=k)
        knn_original.fit(X_train_scaled, y_train)
        y_pred_original = knn_original.predict(X_test_scaled)
        accuracy_original = metrics.accuracy_score(y_test, y_pred_original)
        
        # PCA 데이터 (1차원)
        knn_pca1 = KNeighborsClassifier(n_neighbors=k)
        knn_pca1.fit(X_train_pca[:, :1], y_train)
        y_pred_pca1 = knn_pca1.predict(X_test_pca[:, :1])
        accuracy_pca1 = metrics.accuracy_score(y_test, y_pred_pca1)
        
        # PCA 데이터 (2차원)
        knn_pca2 = KNeighborsClassifier(n_neighbors=k)
        knn_pca2.fit(X_train_pca[:, :2], y_train)
        y_pred_pca2 = knn_pca2.predict(X_test_pca[:, :2])
        accuracy_pca2 = metrics.accuracy_score(y_test, y_pred_pca2)
        
        # 결과 저장
        results = pd.concat([results, pd.DataFrame({
            'random_seed': [seed],
            'k': [k],
            'original_accuracy': [accuracy_original],
            'pca1_accuracy': [accuracy_pca1],
            'pca2_accuracy': [accuracy_pca2]
        })], ignore_index=True)
        
        if k == 5:  # 대표적인 k 값에 대해서만 출력
            print(f"k={k}일 때: 원본 정확도={accuracy_original:.4f}, PCA(1차원)={accuracy_pca1:.4f}, PCA(2차원)={accuracy_pca2:.4f}")

# 결과 집계
# 각 랜덤 시드와 k 값 조합에 대한 평균 정확도 계산
summary = results.groupby('k').agg({
    'original_accuracy': ['mean', 'std'],
    'pca1_accuracy': ['mean', 'std'],
    'pca2_accuracy': ['mean', 'std']
})

print("\n각 k 값에 대한 평균 정확도 (모든 랜덤 시드 기준):")
print(summary)

# 최적의 k 값 찾기
best_k_original = summary['original_accuracy']['mean'].idxmax()
best_k_pca1 = summary['pca1_accuracy']['mean'].idxmax()
best_k_pca2 = summary['pca2_accuracy']['mean'].idxmax()

print(f"\n최적의 k 값 (원본 데이터): {best_k_original}, 평균 정확도: {summary['original_accuracy']['mean'][best_k_original]:.4f}")
print(f"최적의 k 값 (PCA 1차원): {best_k_pca1}, 평균 정확도: {summary['pca1_accuracy']['mean'][best_k_pca1]:.4f}")
print(f"최적의 k 값 (PCA 2차원): {best_k_pca2}, 평균 정확도: {summary['pca2_accuracy']['mean'][best_k_pca2]:.4f}")

# 그래프 그리기
plt.figure(figsize=(12, 8))

# 원본 데이터 정확도 그래프
plt.subplot(2, 1, 1)
for seed in random_seeds:
    seed_data = results[results['random_seed'] == seed]
    plt.plot(seed_data['k'], seed_data['original_accuracy'], 'o-', alpha=0.2, label=f'원본 (seed={seed})')

mean_original = summary['original_accuracy']['mean']
std_original = summary['original_accuracy']['std']
plt.plot(k_values, mean_original, 'o-', ms = 10, color='blue', linewidth=3, label='원본 (평균)')

mean_pca2 = summary['pca2_accuracy']['mean']
std_pca2 = summary['pca2_accuracy']['std']
plt.plot(k_values, mean_pca2, 's-', ms = 10, color='red', linewidth=3, label='PCA 2차원 (평균)')

plt.xlabel('이웃 수 (k)', fontsize=12)
plt.ylabel('테스트 정확도', fontsize=12)
plt.title('여러 랜덤 시드에 대한 KNN 성능 비교: 원본 vs. PCA(2차원)', fontsize=14)
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(k_values[0]-1, k_values[-1]+6)

# PCA 1차원 vs 2차원 비교
plt.subplot(2, 1, 2)
mean_pca1 = summary['pca1_accuracy']['mean']
std_pca1 = summary['pca1_accuracy']['std']
plt.plot(k_values, mean_pca1, '^-', ms = 10, color='green', linewidth=3, label='PCA 1차원 (평균)')

plt.plot(k_values, mean_pca2, 's-', ms = 10, color='red', linewidth=3, label='PCA 2차원 (평균)')

plt.plot(k_values, mean_original, 'o-', ms = 10, color='blue', linewidth=3, label='원본 (평균)')


plt.xlabel('이웃 수 (k)', fontsize=12)
plt.ylabel('테스트 정확도', fontsize=12)
plt.title('여러 랜덤 시드에 대한 KNN 성능 비교: PCA 차원 비교', fontsize=14)
plt.grid(True)
plt.legend(loc='upper right')
plt.xlim(k_values[0]-1, k_values[-1]+6)
plt.tight_layout()
plt.savefig('knn_performance_multiple_seeds.png', dpi=300)
plt.show()
