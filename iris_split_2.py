import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# 데이터 로드
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = ['꽃받침 길이', '꽃받침 너비', '꽃잎 길이', '꽃잎 너비']
target_names = ['세토사', '버시컬러', '버지니카']

# 한글 폰트 설정
font_path = 'Maplestory Light.ttf'

font = fm.FontProperties(fname=font_path)
fm.fontManager.addfont(font_path)
font_name = font.get_name()
plt.rc('font', family=font_name)
# 마이너스 기호 깨짐 방지
plt.rc('axes', unicode_minus=False)  


# 데이터 분할 - 학습용(80%)과 테스트용(20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=55)

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
print("각 주성분의 설명된 분산 비율:")
for i, ratio in enumerate(explained_variance):
    print(f"PC{i+1}: {ratio:.4f}")

print(f"PC1 + PC2의 누적 설명 분산: {explained_variance[0] + explained_variance[1]:.4f}")

# 그래프 설정 - 3x4 그리드로 변경
plt.figure(figsize=(15, 12))  # 세로 크기 늘림

# 원본 특성 히스토그램
for i in range(4):
    plt.subplot(3, 4, i+1)  # 3행 4열로 변경
    for species in range(3):
        plt.hist(X_train[y_train == species, i], alpha=0.5, bins=15, 
                 label=target_names[species])
    plt.xlabel(feature_names[i], fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    plt.legend(loc='upper right')
    plt.title(f'원본 특성: {feature_names[i]}', fontsize=14)

# PCA 특성 히스토그램
for i in range(2):
    plt.subplot(3, 4, i+5)  # 위치 조정
    for species in range(3):
        plt.hist(X_train_pca[y_train == species, i], alpha=0.5, bins=15,
                 label=target_names[species])
    plt.xlabel(f'PC{i+1}', fontsize=12)
    plt.ylabel('빈도', fontsize=12)
    plt.legend(loc='upper right')
    plt.title(f'PCA 특성: PC{i+1}', fontsize=14)

# 원본 특성 산점도 (첫 두 특성)
plt.subplot(3, 4, 9)  # 위치 조정
for species in range(3):
    plt.scatter(X_train[y_train == species, 0], X_train[y_train == species, 1],
                label=target_names[species], alpha=0.7)
plt.xlabel(feature_names[0], fontsize=12)
plt.ylabel(feature_names[1], fontsize=12)
plt.legend()
plt.title('원본 특성 산점도 (첫 두 특성)', fontsize=14)

# 원본 특성 산점도 (세 번째, 네 번째 특성) - 추가
plt.subplot(3, 4, 10)
for species in range(3):
    plt.scatter(X_train[y_train == species, 2], X_train[y_train == species, 3],
                label=target_names[species], alpha=0.7)
plt.xlabel(feature_names[2], fontsize=12)
plt.ylabel(feature_names[3], fontsize=12)
plt.legend()
plt.title('원본 특성 산점도 (세 번째, 네 번째 특성)', fontsize=14)

# PCA 특성 산점도 (PC1, PC2)
plt.subplot(3, 4, 11)  # 위치 조정
for species in range(3):
    plt.scatter(X_train_pca[y_train == species, 0], X_train_pca[y_train == species, 1],
                label=target_names[species], alpha=0.7)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.legend()
plt.title('PCA 특성 산점도 (PC1, PC2)', fontsize=14)

plt.tight_layout()
plt.savefig('iris_histograms_train_data.png', dpi=300)
plt.show()

# KNN 모델 성능 평가 함수
def evaluate_knn_model(train_data, test_data, train_labels, test_labels, k_values, title):
    """다양한 k값에 대해 KNN 모델을 학습하고 테스트 정확도를 반환"""
    accuracies = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_data, train_labels)
        y_pred = knn.predict(test_data)
        accuracy = metrics.accuracy_score(test_labels, y_pred)
        accuracies.append(accuracy)
        print(f"{title}: k={k}일 때 테스트 정확도: {accuracy:.4f}")
        
    return accuracies

# 다양한 k값에 대한 모델 평가
k_values = range(1, 30, 2)

# 원본 데이터와 PCA 변환 데이터에 대한 KNN 모델 평가
print("\n원본 데이터에 대한 KNN 모델 평가:")
original_accuracies = evaluate_knn_model(
    X_train_scaled, X_test_scaled, y_train, y_test, k_values, "원본 데이터")

print("\nPCA 변환 데이터(PC1, PC2)에 대한 KNN 모델 평가:")
pca2_accuracies = evaluate_knn_model(
    X_train_pca[:, :2], X_test_pca[:, :2], y_train, y_test, k_values, "PCA 2차원")

# 성능 비교 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(k_values, original_accuracies, 'o-', label='원본 특성 (4차원)')
plt.plot(k_values, pca2_accuracies, 's-', label='PCA 특성 (2차원)')
plt.xlabel('이웃 수 (k)', fontsize=12)
plt.ylabel('테스트 정확도', fontsize=12)
plt.title('KNN 성능 비교: 원본 vs. PCA', fontsize=14)
plt.legend()
plt.grid(True)
plt.xticks(k_values)
plt.savefig('knn_performance_comparison.png', dpi=300)
plt.show()

# 최적의 k값과 최대 정확도 출력
max_original_acc = max(original_accuracies)
max_original_k = k_values[original_accuracies.index(max_original_acc)]

max_pca_acc = max(pca2_accuracies)
max_pca_k = k_values[pca2_accuracies.index(max_pca_acc)]

print(f"\n원본 데이터 최대 정확도: {max_original_acc:.4f} (k={max_original_k})")
print(f"PCA 데이터(2차원) 최대 정확도: {max_pca_acc:.4f} (k={max_pca_k})")

# # 혼동 행렬(Confusion Matrix) 분석
# def analyze_confusion_matrix(train_data, test_data, train_labels, test_labels, best_k, title):
#     """최적의 k값에 대한 혼동 행렬 분석"""
#     knn = KNeighborsClassifier(n_neighbors=best_k)
#     knn.fit(train_data, train_labels)
#     y_pred = knn.predict(test_data)
    
#     cm = metrics.confusion_matrix(test_labels, y_pred)
    
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title(f'{title} 혼동 행렬 (k={best_k})', fontsize=14)
#     plt.colorbar()
    
#     # 축 레이블 설정
#     classes = target_names
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
#     plt.yticks(tick_marks, classes, fontsize=12)
    
#     # 혼동 행렬 내 값 표시
#     fmt = 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             plt.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
    
#     plt.tight_layout()
#     plt.ylabel('실제 클래스', fontsize=12)
#     plt.xlabel('예측 클래스', fontsize=12)
#     plt.savefig(f'confusion_matrix_{title.replace(" ", "_").lower()}.png', dpi=300)
#     plt.show()
    
#     # 분류 보고서 출력
#     print(f"\n{title} 분류 보고서 (k={best_k}):")
#     print(metrics.classification_report(test_labels, y_pred, target_names=target_names))

# # 최적의 k값에 대한 혼동 행렬 분석
# analyze_confusion_matrix(X_train_scaled, X_test_scaled, y_train, y_test, max_original_k, "원본 데이터")
# analyze_confusion_matrix(X_train_pca[:, :2], X_test_pca[:, :2], y_train, y_test, max_pca_k, "PCA 데이터")