import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

def visualize_ensemble_results(result_df, max_samples=1000, figsize=(20, 18)):
    """
    Усовершенствованная визуализация результатов ансамблевого детектора аномалий.

    Parameters:
    -----------
    result_df : pandas.DataFrame
        Результаты работы метода predict() ансамблевого детектора
    max_samples : int
        Максимальное количество точек для отображения
    figsize : tuple
        Размер фигуры для визуализации
    """
    # Проверяем наличие аномалий
    anomaly_count = result_df['predicted_anomaly'].sum()

    if anomaly_count == 0:
        print("Аномалии не обнаружены. Визуализация невозможна.")
        return

    # Стратифицированная выборка, если данных слишком много
    if len(result_df) > max_samples:
        # Определяем размер выборки для каждого класса
        normal_size = min(int(max_samples * 0.7), (result_df['predicted_anomaly'] == 0).sum())
        anomaly_size = min(int(max_samples * 0.3), anomaly_count)

        # Стратифицированная выборка
        normal_samples = result_df[result_df['predicted_anomaly'] == 0].sample(normal_size, random_state=42)
        anomaly_samples = result_df[result_df['predicted_anomaly'] == 1].sample(anomaly_size, random_state=42)

        # Объединяем выборки
        vis_df = pd.concat([normal_samples, anomaly_samples])
    else:
        vis_df = result_df.copy()

    # Создаем фигуру с сеткой для различных визуализаций
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 3, figure=plt.gcf())

    # Определяем, есть ли в данных истинные метки
    has_true_labels = 'is_anomaly' in vis_df.columns

    # 1. Распределение аномальных оценок
    ax1 = plt.subplot(gs[0, 0])

    try:
        # Используем histplot с KDE, если есть разнообразие в данных
        if vis_df['anomaly_score'].nunique() > 3:
            sns.histplot(data=vis_df, x='anomaly_score', hue='predicted_anomaly',
                         palette={0: 'blue', 1: 'red'}, bins=30, kde=True, ax=ax1)
        else:
            # Если все значения почти одинаковые, используем только гистограмму
            sns.histplot(data=vis_df, x='anomaly_score', hue='predicted_anomaly',
                         palette={0: 'blue', 1: 'red'}, bins=10, kde=False, ax=ax1)
    except Exception as e:
        print(f"Не удалось построить распределение аномальных оценок: {e}")
        ax1.text(0.5, 0.5, 'Невозможно построить график распределения',
                 horizontalalignment='center', verticalalignment='center')

    ax1.set_title('Распределение аномальных оценок')
    ax1.set_xlabel('Аномальная оценка')
    ax1.set_ylabel('Количество')

    # 2. Матрица ошибок (только если есть истинные метки)
    ax2 = plt.subplot(gs[0, 1])

    if has_true_labels:
        try:
            cm = confusion_matrix(vis_df['is_anomaly'], vis_df['predicted_anomaly'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Норма', 'Аномалия'],
                        yticklabels=['Норма', 'Аномалия'], ax=ax2)

            # Вычисляем метрики
            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)

            # Добавляем метрики на график
            metrics_text = f"Accuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\n"
            metrics_text += f"Recall: {recall:.4f}\nF1 Score: {f1:.4f}"

            # Расположение текста справа от матрицы ошибок
            ax2.text(2.5, 0.5, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

            ax2.set_title('Матрица ошибок')
            ax2.set_xlabel('Предсказание')
            ax2.set_ylabel('Истинное значение')
        except Exception as e:
            print(f"Не удалось построить матрицу ошибок: {e}")
            ax2.text(0.5, 0.5, 'Невозможно построить матрицу ошибок',
                     horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, 'Истинные метки аномалий недоступны',
                 horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)

    # 3. Типы аномалий
    ax3 = plt.subplot(gs[0, 2])

    if 'anomaly_type' in vis_df.columns:
        try:
            # Подсчет количества аномалий каждого типа
            anomaly_types = vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_type'].value_counts()

            if not anomaly_types.empty:
                # Создаем горизонтальную столбчатую диаграмму
                anomaly_types.plot(kind='barh', ax=ax3, color='coral')
                ax3.set_title('Типы обнаруженных аномалий')
                ax3.set_xlabel('Количество')
                ax3.set_ylabel('Тип аномалии')
            else:
                ax3.text(0.5, 0.5, 'Нет данных о типах аномалий',
                         horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        except Exception as e:
            print(f"Не удалось построить график типов аномалий: {e}")
            ax3.text(0.5, 0.5, 'Невозможно построить график типов аномалий',
                     horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
    else:
        anomalies = vis_df[vis_df['predicted_anomaly'] == 1]

        # Попытка определить типы аномалий на основе доступных признаков
        anomaly_types = []

        if 'bytes' in vis_df.columns and 'packets' in vis_df.columns:
            try:
                # Аномалии объема
                high_volume = (vis_df['bytes'] > vis_df['bytes'].quantile(0.95)) & (vis_df['predicted_anomaly'] == 1)
                anomaly_types.append(('Большой объем трафика', high_volume.sum()))
            except Exception as e:
                print(f"Не удалось определить аномалии объема: {e}")

        if 'dst_port' in vis_df.columns:
            try:
                # Аномалии портов
                unusual_ports = [6667, 31337, 4444, 9001, 1337, 8080]
                unusual_port = (vis_df['dst_port'].isin(unusual_ports)) & (vis_df['predicted_anomaly'] == 1)
                anomaly_types.append(('Необычные порты', unusual_port.sum()))
            except Exception as e:
                print(f"Не удалось определить аномалии портов: {e}")

        # Построение графика типов аномалий, если есть данные
        if anomaly_types:
            try:
                anomaly_df = pd.DataFrame(anomaly_types, columns=['type', 'count'])
                anomaly_df = anomaly_df.sort_values('count', ascending=False)
                if not anomaly_df.empty and anomaly_df['count'].sum() > 0:
                    anomaly_df.plot(kind='barh', x='type', y='count', legend=False, ax=ax3, color='coral')
                    ax3.set_title('Типы обнаруженных аномалий')
                    ax3.set_xlabel('Количество')
                    ax3.set_ylabel('Тип аномалии')
                else:
                    ax3.text(0.5, 0.5, 'Недостаточно данных для анализа типов аномалий',
                             horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
            except Exception as e:
                print(f"Не удалось построить график типов аномалий: {e}")
                ax3.text(0.5, 0.5, 'Ошибка при построении графика типов аномалий',
                         horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, 'Недостаточно данных для определения типов аномалий',
                     horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)

    # 4. Временной ряд аномалий (если есть timestamp)
    ax4 = plt.subplot(gs[1, :2])

    if 'timestamp' in vis_df.columns:
        try:
            # Конвертируем в datetime, если нужно
            if vis_df['timestamp'].dtype != 'datetime64[ns]':
                vis_df['timestamp'] = pd.to_datetime(vis_df['timestamp'])

            # Создаем временной ряд с оценками аномальности
            scatter = ax4.scatter(vis_df['timestamp'], vis_df['anomaly_score'],
                                 c=vis_df['predicted_anomaly'], cmap='coolwarm', alpha=0.7)

            # Добавляем линию тренда
            try:
                from scipy.signal import savgol_filter

                # Сортируем по времени для корректного сглаживания
                temp_df = vis_df.sort_values('timestamp')

                # Сглаживание Савицкого-Голея (если достаточно точек)
                if len(temp_df) > 10:
                    window_size = min(15, len(temp_df) // 2 * 2 - 1)  # Должно быть нечетным
                    yhat = savgol_filter(temp_df['anomaly_score'], window_size, 3)
                    ax4.plot(temp_df['timestamp'], yhat, 'k-', lw=2, alpha=0.5)
            except Exception as e:
                print(f"Не удалось построить линию тренда: {e}")

            # Форматирование графика
            ax4.set_title('Временной ряд аномальных оценок')
            ax4.set_xlabel('Время')
            ax4.set_ylabel('Аномальная оценка')
            ax4.legend(*scatter.legend_elements(), title="Аномалия")

            # Автоматическое форматирование дат
            plt.gcf().autofmt_xdate()
        except Exception as e:
            print(f"Не удалось построить временной ряд аномалий: {e}")
            ax4.text(0.5, 0.5, 'Невозможно построить временной ряд аномалий',
                     horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
    else:
        # Если нет временной метки, просто показываем аномальные оценки по индексу
        try:
            scatter = ax4.scatter(range(len(vis_df)), vis_df['anomaly_score'],
                                 c=vis_df['predicted_anomaly'], cmap='coolwarm', alpha=0.7)

            ax4.set_title('Аномальные оценки по индексу')
            ax4.set_xlabel('Индекс')
            ax4.set_ylabel('Аномальная оценка')
            ax4.legend(*scatter.legend_elements(), title="Аномалия")
        except Exception as e:
            print(f"Не удалось построить график аномальных оценок: {e}")
            ax4.text(0.5, 0.5, 'Невозможно построить график аномальных оценок',
                     horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

    # 5. PCA или t-SNE для визуализации в 2D
    ax5 = plt.subplot(gs[1, 2])

    try:
        # Определяем числовые признаки для снижения размерности
        numeric_cols = vis_df.select_dtypes(include=['int64', 'float64']).columns

        # Исключаем технические колонки и метки
        exclude_cols = ['predicted_anomaly', 'anomaly_score', 'is_anomaly']
        exclude_cols.extend([col for col in vis_df.columns if col.startswith('anomaly_') or col.startswith('score_')])

        feature_cols = [col for col in numeric_cols if col not in exclude_cols]

        if len(feature_cols) >= 2:
            # Если признаков много, используем PCA или t-SNE
            if len(feature_cols) > 2:
                # Выбираем метод снижения размерности
                if len(vis_df) < 5000:  # t-SNE медленный для больших выборок
                    # t-SNE для нелинейного снижения размерности
                    reducer = TSNE(n_components=2, random_state=42)
                    vis_title = 't-SNE визуализация'
                else:
                    # PCA для линейного снижения размерности
                    reducer = PCA(n_components=2, random_state=42)
                    vis_title = 'PCA визуализация'

                # Применяем снижение размерности
                embedding = reducer.fit_transform(vis_df[feature_cols])

                # Визуализируем в 2D пространстве
                scatter = ax5.scatter(embedding[:, 0], embedding[:, 1],
                                     c=vis_df['predicted_anomaly'], cmap='coolwarm', alpha=0.7)
            else:
                # Если уже есть 2 признака, просто используем их
                scatter = ax5.scatter(vis_df[feature_cols[0]], vis_df[feature_cols[1]],
                                     c=vis_df['predicted_anomaly'], cmap='coolwarm', alpha=0.7)
                vis_title = f'Визуализация по признакам: {feature_cols[0]} и {feature_cols[1]}'

            ax5.set_title(vis_title)
            ax5.set_xlabel('Компонента 1')
            ax5.set_ylabel('Компонента 2')
            ax5.legend(*scatter.legend_elements(), title="Аномалия")
        else:
            ax5.text(0.5, 0.5, 'Недостаточно числовых признаков для визуализации',
                     horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
    except Exception as e:
        print(f"Не удалось построить 2D визуализацию: {e}")
        ax5.text(0.5, 0.5, 'Невозможно построить 2D визуализацию',
                 horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)

    # 6. ROC и Precision-Recall кривые (если есть истинные метки)
    ax6 = plt.subplot(gs[2, 0])
    ax7 = plt.subplot(gs[2, 1])

    if has_true_labels:
        try:
            # ROC кривая
            fpr, tpr, _ = roc_curve(vis_df['is_anomaly'], vis_df['anomaly_score'])
            roc_auc = auc(fpr, tpr)

            ax6.plot(fpr, tpr, lw=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
            ax6.plot([0, 1], [0, 1], 'k--', lw=1)
            ax6.set_xlim([0.0, 1.0])
            ax6.set_ylim([0.0, 1.05])
            ax6.set_xlabel('False Positive Rate')
            ax6.set_ylabel('True Positive Rate')
            ax6.set_title('ROC кривая')
            ax6.legend(loc="lower right")

            # Precision-Recall кривая
            precision, recall, _ = precision_recall_curve(vis_df['is_anomaly'], vis_df['anomaly_score'])
            pr_auc = auc(recall, precision)

            ax7.plot(recall, precision, lw=2, label=f'PR кривая (AUC = {pr_auc:.3f})')
            ax7.set_xlim([0.0, 1.0])
            ax7.set_ylim([0.0, 1.05])
            ax7.set_xlabel('Recall')
            ax7.set_ylabel('Precision')
            ax7.set_title('Precision-Recall кривая')
            ax7.legend(loc="lower left")
        except Exception as e:
            print(f"Не удалось построить ROC и PR кривые: {e}")
            ax6.text(0.5, 0.5, 'Невозможно построить ROC кривую',
                     horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
            ax7.text(0.5, 0.5, 'Невозможно построить PR кривую',
                     horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes)
    else:
        ax6.text(0.5, 0.5, 'Истинные метки аномалий недоступны для ROC',
                 horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
        ax7.text(0.5, 0.5, 'Истинные метки аномалий недоступны для PR кривой',
                 horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes)

    # 7. Сравнение детекторов (если информация доступна)
    ax8 = plt.subplot(gs[2, 2])

    # Проверка наличия данных от разных детекторов
    detector_cols = [col for col in vis_df.columns if col.startswith('score_')]

    if detector_cols:
        try:
            # Сравнение производительности разных детекторов
            detector_names = [col.replace('score_', '') for col in detector_cols]

            # Если есть истинные метки, вычисляем AUC
            if has_true_labels:
                aucs = []
                for col in detector_cols:
                    try:
                        detector_auc = roc_auc_score(vis_df['is_anomaly'], vis_df[col])
                        aucs.append(detector_auc)
                    except:
                        aucs.append(0)

                # Создаем датафрейм для визуализации
                detector_df = pd.DataFrame({
                    'Detector': detector_names,
                    'AUC': aucs
                })

                # Сортируем по AUC
                detector_df = detector_df.sort_values('AUC', ascending=False)

                # Визуализируем
                sns.barplot(x='AUC', y='Detector', data=detector_df, ax=ax8, palette='viridis')
                ax8.set_title('Сравнение детекторов по AUC')
                ax8.set_xlabel('AUC')
                ax8.set_ylabel('Детектор')
            else:
                # Если нет истинных меток, просто показываем среднюю оценку аномальности
                avg_scores = []
                for col in detector_cols:
                    avg_scores.append(vis_df[col].mean())

                # Создаем датафрейм для визуализации
                detector_df = pd.DataFrame({
                    'Detector': detector_names,
                    'Avg Score': avg_scores
                })

                # Сортируем по средней оценке
                detector_df = detector_df.sort_values('Avg Score', ascending=False)

                # Визуализируем
                sns.barplot(x='Avg Score', y='Detector', data=detector_df, ax=ax8, palette='viridis')
                ax8.set_title('Сравнение детекторов по средней оценке аномалий')
                ax8.set_xlabel('Средняя оценка')
                ax8.set_ylabel('Детектор')
        except Exception as e:
            print(f"Не удалось построить сравнение детекторов: {e}")
            ax8.text(0.5, 0.5, 'Невозможно построить сравнение детекторов',
                     horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
    else:
        ax8.text(0.5, 0.5, 'Нет данных от разных детекторов для сравнения',
                 horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)

    # Подгоняем лейауты и отображаем график
    plt.tight_layout()
    plt.show()

    # Если аномалии обнаружены, выводим дополнительную статистику
    if anomaly_count > 0:
        print(f"\nОбнаружено аномалий: {anomaly_count} ({anomaly_count/len(result_df)*100:.2f}%)")

        if 'anomaly_type' in result_df.columns:
            print("\nРаспределение типов аномалий:")
            type_counts = result_df[result_df['predicted_anomaly'] == 1]['anomaly_type'].value_counts()
            for anomaly_type, count in type_counts.items():
                print(f"  {anomaly_type}: {count} ({count/anomaly_count*100:.2f}%)")

        if has_true_labels:
            # Вычисляем метрики эффективности
            tn, fp, fn, tp = confusion_matrix(result_df['is_anomaly'], result_df['predicted_anomaly']).ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print("\nМетрики эффективности:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  False Positive Rate: {fp/(fp+tn):.4f}")
            print(f"  False Negative Rate: {fn/(fn+tp):.4f}")


def visualize_sequence_analysis(sequence_detector, data, max_sequences=10, figsize=(15, 10)):
    """
    Визуализация анализа последовательностей для детектора SequenceAnomalyDetector.

    Parameters:
    -----------
    sequence_detector : SequenceAnomalyDetector
        Обученный детектор последовательностей
    data : pandas.DataFrame
        Данные для анализа
    max_sequences : int
        Максимальное количество последовательностей для отображения
    figsize : tuple
        Размер фигуры для визуализации
    """
    if not hasattr(sequence_detector, 'transition_probs') or not sequence_detector.transition_probs:
        print("Детектор последовательностей не обучен или не содержит информации о переходах.")
        return

    # Предобработка данных
    preprocessed_data = sequence_detector.preprocess_data(data)

    # Получаем последовательности состояний
    states = preprocessed_data['state'].tolist()

    # Получаем предсказания
    predictions = sequence_detector.predict(data)

    # Выделяем наиболее аномальные последовательности
    anomaly_indices = predictions[predictions['predicted_anomaly'] == 1].index

    if len(anomaly_indices) == 0:
        print("Аномальные последовательности не обнаружены.")
        return

    # Сортируем аномалии по убыванию оценки аномальности
    sorted_anomalies = predictions.loc[anomaly_indices].sort_values('anomaly_score', ascending=False)

    # Выбираем топ-N аномальных последовательностей
    top_anomalies = sorted_anomalies.head(max_sequences)

    # Создаем фигуру
    plt.figure(figsize=figsize)

    # 1. Визуализация графа переходов
    plt.subplot(2, 2, 1)

    # Создаем граф переходов
    import networkx as nx
    G = nx.DiGraph()

    # Добавляем узлы и ребра
    for current_state, transitions in sequence_detector.transition_probs.items():
        for next_state, prob in transitions.items():
            if prob > 0.01:  # Отображаем только значимые переходы
                G.add_edge(current_state, next_state, weight=prob)

    # Ограничиваем размер графа для наглядности
    if len(G.nodes()) > 25:
        # Оставляем только узлы с наибольшей степенью
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:25]
        nodes_to_keep = [node for node, _ in top_nodes]
        G = G.subgraph(nodes_to_keep)

    # Определяем веса для ребер и размеры узлов
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    node_sizes = [G.degree(node) * 100 for node in G.nodes()]

    # Рисуем граф
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx(G, pos, with_labels=False, node_size=node_sizes,
                     node_color="skyblue", alpha=0.7, edge_color="gray",
                     width=edge_weights, arrowsize=10)

    # Добавляем метки только для ключевых узлов
    if len(G.nodes()) <= 10:
        nx.draw_networkx_labels(G, pos)
    else:
        # Метки только для узлов с наибольшей степенью
        top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
        nodes_to_label = {node: node for node, _ in top_nodes}
        nx.draw_networkx_labels(G, pos, labels=nodes_to_label)

    plt.title('Граф переходов между состояниями')
    plt.axis('off')

    # 2. Тепловая карта вероятностей переходов
    plt.subplot(2, 2, 2)

    # Получаем топ-N состояний для тепловой карты
    all_states = set()
    for state in sequence_detector.transition_probs:
        all_states.add(state)
        all_states.update(sequence_detector.transition_probs[state].keys())

    # Выбираем наиболее важные состояния
    if len(all_states) <= 15:
        important_states = list(all_states)
    else:
        # Выбираем состояния с наибольшим количеством переходов
        state_freq = {}
        for state in all_states:
            freq = 0
            # Подсчитываем входящие переходы
            for s, transitions in sequence_detector.transition_probs.items():
                if state in transitions:
                    freq += 1
            # Подсчитываем исходящие переходы
            if state in sequence_detector.transition_probs:
                freq += len(sequence_detector.transition_probs[state])

            state_freq[state] = freq

        # Выбираем топ-15 состояний
        important_states = sorted(state_freq.items(), key=lambda x: x[1], reverse=True)[:15]
        important_states = [state for state, _ in important_states]

    # Создаем матрицу вероятностей переходов
    transition_matrix = np.zeros((len(important_states), len(important_states)))

    for i, state_from in enumerate(important_states):
        for j, state_to in enumerate(important_states):
            if state_from in sequence_detector.transition_probs and state_to in sequence_detector.transition_probs[state_from]:
                transition_matrix[i, j] = sequence_detector.transition_probs[state_from][state_to]

    # Создаем тепловую карту
    sns.heatmap(transition_matrix, annot=False, cmap='Blues',
                xticklabels=important_states, yticklabels=important_states)
    plt.title('Тепловая карта вероятностей переходов')
    plt.xlabel('Следующее состояние')
    plt.ylabel('Текущее состояние')

    # Скрываем метки состояний, если их слишком много
    if len(important_states) > 8:
        plt.xticks([])
        plt.yticks([])

    # 3. Распределение аномальных оценок
    plt.subplot(2, 2, 3)

    # Отображаем гистограмму аномальных оценок
    sns.histplot(data=predictions, x='anomaly_score', hue='predicted_anomaly',
                kde=True, palette={0: 'blue', 1: 'red'})
    plt.title('Распределение аномальных оценок')
    plt.xlabel('Аномальная оценка')
    plt.ylabel('Количество')

    # 4. Топ аномальных последовательностей
    plt.subplot(2, 2, 4)

    # Создаем таблицу для отображения топ аномальных последовательностей
    cell_text = []

    for i, (idx, row) in enumerate(top_anomalies.iterrows()):
        # Получаем последовательность вокруг аномальной точки
        seq_start = max(0, idx - sequence_detector.sequence_length // 2)
        seq_end = min(len(states), idx + sequence_detector.sequence_length // 2 + 1)

        # Ограничиваем длину отображаемой последовательности
        sequence = states[seq_start:seq_end]
        if len(sequence) > 5:
            sequence = sequence[:2] + ['...'] + sequence[-2:]

        sequence_str = ' -> '.join(sequence)

        # Если строка слишком длинная, сокращаем
        if len(sequence_str) > 30:
            sequence_str = sequence_str[:27] + '...'

        cell_text.append([i+1, sequence_str, f"{row['anomaly_score']:.4f}"])

    # Если нет аномальных последовательностей, показываем сообщение
    if not cell_text:
        plt.text(0.5, 0.5, 'Нет данных о аномальных последовательностях',
                 horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    else:
        # Создаем таблицу
        table = plt.table(
            cellText=cell_text,
            colLabels=['#', 'Последовательность', 'Оценка аномальности'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        plt.axis('off')
        plt.title('Топ аномальных последовательностей')

    # Подгоняем лейауты и отображаем график
    plt.tight_layout()
    plt.show()

    # Выводим дополнительную информацию
    print(f"Всего уникальных состояний: {len(sequence_detector.frequency_dict)}")
    print(f"Всего переходов между состояниями: {sum(len(transitions) for transitions in sequence_detector.transition_probs.values())}")
    print(f"Обнаружено аномальных последовательностей: {len(anomaly_indices)} ({len(anomaly_indices)/len(predictions)*100:.2f}%)")
