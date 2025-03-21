"""Модуль для визуализации результатов анализа аномалий.

Предоставляет классы для создания графиков, диаграмм и других визуальных
представлений результатов работы детекторов аномалий.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import matplotlib.gridspec as gridspec
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class BaseVisualizer:
    """
    Базовый класс для всех визуализаторов.

    Обеспечивает общую функциональность для визуализации результатов анализа аномалий,
    включая настройку графиков, обработку ошибок и экспорт результатов.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 6), style: str = 'whitegrid',
                 palette: str = 'viridis', dpi: int = 100, output_dir: str = "visualizations"):
        """
        Инициализация базового визуализатора.

        Parameters:
        -----------
        figsize : Tuple[int, int]
            Размер графика (ширина, высота) в дюймах
        style : str
            Стиль графика для seaborn (whitegrid, darkgrid, white, dark, ticks)
        palette : str
            Цветовая палитра для графиков
        dpi : int
            Разрешение графика в DPI
        output_dir : str
            Директория для сохранения графиков
        """
        self.figsize = figsize
        self.style = style
        self.palette = palette
        self.dpi = dpi
        self.output_dir = output_dir

        # Настройка стиля
        sns.set_style(style)
        sns.set_palette(palette)

        # Создание директории для сохранения, если не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def setup_figure(self, title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Создание и настройка фигуры и осей.

        Parameters:
        -----------
        title : Optional[str]
            Заголовок графика

        Returns:
        --------
        Tuple[plt.Figure, plt.Axes]
            Кортеж (фигура, оси)
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if title:
            ax.set_title(title, fontsize=14, pad=20)

        return fig, ax

    def save_figure(self, fig: plt.Figure, filename: Optional[str] = None,
                   formats: List[str] = ['png', 'pdf']) -> List[str]:
        """
        Сохранение графика в файл(ы).

        Parameters:
        -----------
        fig : plt.Figure
            Фигура для сохранения
        filename : Optional[str]
            Имя файла (без расширения)
        formats : List[str]
            Форматы для сохранения

        Returns:
        --------
        List[str]
            Список путей к сохраненным файлам
        """
        # Генерация имени файла, если не указано
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"visualization_{timestamp}"

        # Сохранение в разных форматах
        saved_files = []
        for fmt in formats:
            filepath = os.path.join(self.output_dir, f"{filename}.{fmt}")
            fig.savefig(filepath, format=fmt, bbox_inches='tight')
            saved_files.append(filepath)
            print(f"График сохранен в {filepath}")

        return saved_files

    def handle_plot_error(self, ax: plt.Axes, error_message: str, exception: Optional[Exception] = None) -> None:
        """
        Обработка ошибок при построении графиков.

        Parameters:
        -----------
        ax : plt.Axes
            Оси для отображения сообщения об ошибке
        error_message : str
            Сообщение для отображения на графике
        exception : Optional[Exception]
            Исключение, вызвавшее ошибку (для логирования)
        """
        if exception:
            print(f"Ошибка при построении графика: {str(exception)}")

        # Очищаем оси
        ax.clear()

        # Отображаем сообщение об ошибке
        ax.text(0.5, 0.5, error_message,
                horizontalalignment='center', verticalalignment='center', 
                transform=ax.transAxes, fontsize=10)
        
        # Отключаем оси
        ax.set_xticks([])
        ax.set_yticks([])


class EnsembleVisualizer(BaseVisualizer):
    """
    Визуализатор результатов ансамблевого детектора аномалий.

    Позволяет создавать различные визуализации результатов работы
    ансамблевого детектора, включая распределения оценок, матрицы ошибок,
    типы аномалий, временные ряды и другие.
    """

    def __init__(self, max_samples: int = 1000, **kwargs):
        """
        Инициализация визуализатора ансамблевых результатов.

        Parameters:
        -----------
        max_samples : int
            Максимальное количество точек для отображения
        **kwargs : dict
            Дополнительные параметры для базового визуализатора
        """
        super().__init__(**kwargs)
        self.max_samples = max_samples

    def visualize_ensemble_results(self, result_df: pd.DataFrame, show: bool = True,
                                 save: bool = False, filename: Optional[str] = None) -> plt.Figure:
        """
        Комплексная визуализация результатов ансамблевого детектора аномалий.

        Parameters:
        -----------
        result_df : pandas.DataFrame
            Результаты работы метода predict() ансамблевого детектора
        show : bool
            Отображать график после создания
        save : bool
            Сохранять график в файлы
        filename : Optional[str]
            Имя файла для сохранения (без расширения)

        Returns:
        --------
        plt.Figure
            Созданная фигура с визуализациями
        """
        # Проверяем наличие аномалий
        anomaly_count = result_df['predicted_anomaly'].sum()

        if anomaly_count == 0:
            print("Аномалии не обнаружены. Визуализация невозможна.")
            fig, ax = self.setup_figure("Результаты анализа аномалий")
            self.handle_plot_error(ax, "Аномалии не обнаружены.")
            return fig

        # Подготавливаем данные для визуализации
        vis_df = self._prepare_data_for_visualization(result_df)

        # Определяем, есть ли в данных истинные метки
        has_true_labels = 'is_anomaly' in vis_df.columns

        # Создаем фигуру с сеткой для различных визуализаций
        fig = plt.figure(figsize=(20, 18), dpi=self.dpi)
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # 1. Распределение аномальных оценок
        ax1 = plt.subplot(gs[0, 0])
        self._plot_anomaly_score_distribution(ax1, vis_df)

        # 2. Матрица ошибок (только если есть истинные метки)
        ax2 = plt.subplot(gs[0, 1])
        if has_true_labels:
            self._plot_confusion_matrix(ax2, vis_df)
        else:
            self.handle_plot_error(ax2, "Истинные метки аномалий недоступны")

        # 3. Типы аномалий
        ax3 = plt.subplot(gs[0, 2])
        self._plot_anomaly_types(ax3, vis_df)

        # 4. Временной ряд аномалий (если есть timestamp)
        ax4 = plt.subplot(gs[1, :2])
        self._plot_time_series(ax4, vis_df)

        # 5. PCA или t-SNE для визуализации в 2D
        ax5 = plt.subplot(gs[1, 2])
        self._plot_dimensionality_reduction(ax5, vis_df)

        # 6. ROC кривая
        ax6 = plt.subplot(gs[2, 0])
        if has_true_labels:
            self._plot_roc_curve(ax6, vis_df)
        else:
            self.handle_plot_error(ax6, "Истинные метки аномалий недоступны для ROC")

        # 7. Precision-Recall кривая
        ax7 = plt.subplot(gs[2, 1])
        if has_true_labels:
            self._plot_precision_recall_curve(ax7, vis_df)
        else:
            self.handle_plot_error(ax7, "Истинные метки аномалий недоступны для PR кривой")

        # 8. Сравнение детекторов
        ax8 = plt.subplot(gs[2, 2])
        self._plot_detector_comparison(ax8, vis_df, has_true_labels)

        # Подгоняем лейауты
        plt.tight_layout()
        
        # Отображаем график
        if show:
            plt.show()
            
        # Выводим статистику по аномалиям
        self._print_anomaly_statistics(result_df, has_true_labels)
        
        # Сохраняем график, если нужно
        if save:
            self.save_figure(fig, filename)
            
        return fig

    def _prepare_data_for_visualization(self, result_df: pd.DataFrame) -> pd.DataFrame:
        """
        Подготовка данных для визуализации, включая стратифицированную выборку.

        Parameters:
        -----------
        result_df : pandas.DataFrame
            Исходные результаты

        Returns:
        --------
        pandas.DataFrame
            Подготовленные данные для визуализации
        """
        # Проверяем наличие аномалий
        anomaly_count = result_df['predicted_anomaly'].sum()

        # Стратифицированная выборка, если данных слишком много
        if len(result_df) > self.max_samples:
            # Определяем размер выборки для каждого класса
            normal_size = min(int(self.max_samples * 0.7), (result_df['predicted_anomaly'] == 0).sum())
            anomaly_size = min(int(self.max_samples * 0.3), anomaly_count)

            # Стратифицированная выборка
            normal_samples = result_df[result_df['predicted_anomaly'] == 0].sample(normal_size, random_state=42)
            anomaly_samples = result_df[result_df['predicted_anomaly'] == 1].sample(anomaly_size, random_state=42)

            # Объединяем выборки
            vis_df = pd.concat([normal_samples, anomaly_samples])
        else:
            vis_df = result_df.copy()

        return vis_df

    def _print_anomaly_statistics(self, result_df: pd.DataFrame, has_true_labels: bool) -> None:
        """
        Вывод статистики по обнаруженным аномалиям.

        Parameters:
        -----------
        result_df : pandas.DataFrame
            Данные результатов
        has_true_labels : bool
            Есть ли истинные метки аномалий
        """
        anomaly_count = result_df['predicted_anomaly'].sum()
        
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

    def _plot_anomaly_score_distribution(self, ax: plt.Axes, vis_df: pd.DataFrame) -> None:
        """
        Построение распределения аномальных оценок.

        Parameters:
        -----------
        ax : plt.Axes
            Оси для построения графика
        vis_df : pandas.DataFrame
            Данные для визуализации
        """
        try:
            # Используем histplot с KDE, если есть разнообразие в данных
            if vis_df['anomaly_score'].nunique() > 3:
                sns.histplot(data=vis_df, x='anomaly_score', hue='predicted_anomaly',
                           palette={0: 'blue', 1: 'red'}, bins=30, kde=True, ax=ax)
            else:
                # Если все значения почти одинаковые, используем только гистограмму
                sns.histplot(data=vis_df, x='anomaly_score', hue='predicted_anomaly',
                           palette={0: 'blue', 1: 'red'}, bins=10, kde=False, ax=ax)

            ax.set_title('Распределение аномальных оценок')
            ax.set_xlabel('Аномальная оценка')
            ax.set_ylabel('Количество')
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить график распределения', e)

    def _plot_confusion_matrix(self, ax: plt.Axes, vis_df: pd.DataFrame) -> None:
        """
        Построение матрицы ошибок.

        Parameters:
        -----------
        ax : plt.Axes
            Оси для построения графика
        vis_df : pandas.DataFrame
            Данные для визуализации
        """
        try:
            cm = confusion_matrix(vis_df['is_anomaly'], vis_df['predicted_anomaly'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Норма', 'Аномалия'],
                      yticklabels=['Норма', 'Аномалия'], ax=ax)

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
            ax.text(2.5, 0.5, metrics_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

            ax.set_title('Матрица ошибок')
            ax.set_xlabel('Предсказание')
            ax.set_ylabel('Истинное значение')
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить матрицу ошибок', e)

    def _plot_anomaly_types(self, ax: plt.Axes, vis_df: pd.DataFrame) -> None:
        """Построение графика типов аномалий."""
        if 'anomaly_type' in vis_df.columns:
            try:
                anomaly_types = vis_df[vis_df['predicted_anomaly'] == 1]['anomaly_type'].value_counts()
                if not anomaly_types.empty:
                    anomaly_types.plot(kind='barh', ax=ax, color='coral')
                    ax.set_title('Типы обнаруженных аномалий')
                    ax.set_xlabel('Количество')
                    ax.set_ylabel('Тип аномалии')
                else:
                    self.handle_plot_error(ax, 'Нет данных о типах аномалий')
            except Exception as e:
                self.handle_plot_error(ax, 'Невозможно построить график типов аномалий', e)
        else:
            self._plot_inferred_anomaly_types(ax, vis_df)
    
    def _plot_inferred_anomaly_types(self, ax: plt.Axes, vis_df: pd.DataFrame) -> None:
        """Построение графика предполагаемых типов аномалий."""
        anomaly_types = []
        try:
            if 'bytes' in vis_df.columns and 'packets' in vis_df.columns:
                high_volume = (vis_df['bytes'] > vis_df['bytes'].quantile(0.95)) & (vis_df['predicted_anomaly'] == 1)
                anomaly_types.append(('Большой объем трафика', high_volume.sum()))
            if 'dst_port' in vis_df.columns:
                unusual_ports = [6667, 31337, 4444, 9001, 1337, 8080]
                unusual_port = (vis_df['dst_port'].isin(unusual_ports)) & (vis_df['predicted_anomaly'] == 1)
                anomaly_types.append(('Необычные порты', unusual_port.sum()))
            
            if anomaly_types:
                anomaly_df = pd.DataFrame(anomaly_types, columns=['type', 'count'])
                anomaly_df = anomaly_df.sort_values('count', ascending=False)
                if not anomaly_df.empty and anomaly_df['count'].sum() > 0:
                    anomaly_df.plot(kind='barh', x='type', y='count', legend=False, ax=ax, color='coral')
                    ax.set_title('Типы обнаруженных аномалий')
                    ax.set_xlabel('Количество')
                    ax.set_ylabel('Тип аномалии')
                else:
                    self.handle_plot_error(ax, 'Недостаточно данных для анализа типов аномалий')
            else:
                self.handle_plot_error(ax, 'Недостаточно данных для определения типов аномалий')
        except Exception as e:
            self.handle_plot_error(ax, 'Ошибка при построении графика типов аномалий', e)
    
    def _plot_time_series(self, ax: plt.Axes, vis_df: pd.DataFrame) -> None:
        """Построение временного ряда аномальных оценок."""
        if 'timestamp' in vis_df.columns:
            try:
                if vis_df['timestamp'].dtype != 'datetime64[ns]':
                    vis_df['timestamp'] = pd.to_datetime(vis_df['timestamp'])
                
                scatter = ax.scatter(vis_df['timestamp'], vis_df['anomaly_score'],
                                  c=vis_df['predicted_anomaly'], cmap='coolwarm', alpha=0.7)
                
                try:
                    from scipy.signal import savgol_filter
                    temp_df = vis_df.sort_values('timestamp')
                    if len(temp_df) > 10:
                        window_size = min(15, len(temp_df) // 2 * 2 - 1)
                        yhat = savgol_filter(temp_df['anomaly_score'], window_size, 3)
                        ax.plot(temp_df['timestamp'], yhat, 'k-', lw=2, alpha=0.5)
                except Exception as e:
                    print(f"Не удалось построить линию тренда: {e}")
                
                ax.set_title('Временной ряд аномальных оценок')
                ax.set_xlabel('Время')
                ax.set_ylabel('Аномальная оценка')
                ax.legend(*scatter.legend_elements(), title="Аномалия")
                plt.gcf().autofmt_xdate()
            except Exception as e:
                self.handle_plot_error(ax, 'Невозможно построить временной ряд аномалий', e)
        else:
            try:
                scatter = ax.scatter(range(len(vis_df)), vis_df['anomaly_score'],
                                  c=vis_df['predicted_anomaly'], cmap='coolwarm', alpha=0.7)
                ax.set_title('Аномальные оценки по индексу')
                ax.set_xlabel('Индекс')
                ax.set_ylabel('Аномальная оценка')
                ax.legend(*scatter.legend_elements(), title="Аномалия")
            except Exception as e:
                self.handle_plot_error(ax, 'Невозможно построить график аномальных оценок', e)
    
    def _plot_dimensionality_reduction(self, ax: plt.Axes, vis_df: pd.DataFrame) -> None:
        """Построение визуализации данных в 2D пространстве."""
        try:
            numeric_cols = vis_df.select_dtypes(include=['int64', 'float64']).columns
            exclude_cols = ['predicted_anomaly', 'anomaly_score', 'is_anomaly']
            exclude_cols.extend([col for col in vis_df.columns if col.startswith('anomaly_') or col.startswith('score_')])
            feature_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if len(feature_cols) >= 2:
                if len(feature_cols) > 2:
                    if len(vis_df) < 5000:
                        reducer = TSNE(n_components=2, random_state=42)
                        vis_title = 't-SNE визуализация'
                    else:
                        reducer = PCA(n_components=2, random_state=42)
                        vis_title = 'PCA визуализация'
                    
                    embedding = reducer.fit_transform(vis_df[feature_cols])
                    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                                       c=vis_df['predicted_anomaly'], cmap='coolwarm', alpha=0.7)
                else:
                    scatter = ax.scatter(vis_df[feature_cols[0]], vis_df[feature_cols[1]],
                                       c=vis_df['predicted_anomaly'], cmap='coolwarm', alpha=0.7)
                    vis_title = f'Визуализация по признакам: {feature_cols[0]} и {feature_cols[1]}'
                
                ax.set_title(vis_title)
                ax.set_xlabel('Компонента 1')
                ax.set_ylabel('Компонента 2')
                ax.legend(*scatter.legend_elements(), title="Аномалия")
            else:
                self.handle_plot_error(ax, 'Недостаточно числовых признаков для визуализации')
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить 2D визуализацию', e)
    
    def _plot_roc_curve(self, ax: plt.Axes, vis_df: pd.DataFrame) -> None:
        """Построение ROC-кривой."""
        try:
            fpr, tpr, _ = roc_curve(vis_df['is_anomaly'], vis_df['anomaly_score'])
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, lw=2, label=f'ROC кривая (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC кривая')
            ax.legend(loc="lower right")
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить ROC кривую', e)
    
    def _plot_precision_recall_curve(self, ax: plt.Axes, vis_df: pd.DataFrame) -> None:
        """Построение Precision-Recall кривой."""
        try:
            precision, recall, _ = precision_recall_curve(vis_df['is_anomaly'], vis_df['anomaly_score'])
            pr_auc = auc(recall, precision)
            
            ax.plot(recall, precision, lw=2, label=f'PR кривая (AUC = {pr_auc:.3f})')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall кривая')
            ax.legend(loc="lower left")
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить PR кривую', e)
    
    def _plot_detector_comparison(self, ax: plt.Axes, vis_df: pd.DataFrame, has_true_labels: bool) -> None:
        """Построение сравнения детекторов."""
        detector_cols = [col for col in vis_df.columns if col.startswith('score_')]
        
        if detector_cols:
            try:
                detector_names = [col.replace('score_', '') for col in detector_cols]
                
                if has_true_labels:
                    aucs = []
                    for col in detector_cols:
                        try:
                            detector_auc = roc_auc_score(vis_df['is_anomaly'], vis_df[col])
                            aucs.append(detector_auc)
                        except:
                            aucs.append(0)
                    
                    detector_df = pd.DataFrame({
                        'Detector': detector_names,
                        'AUC': aucs
                    })
                    
                    detector_df = detector_df.sort_values('AUC', ascending=False)
                    sns.barplot(x='AUC', y='Detector', data=detector_df, ax=ax, palette='viridis')
                    ax.set_title('Сравнение детекторов по AUC')
                    ax.set_xlabel('AUC')
                    ax.set_ylabel('Детектор')
                else:
                    avg_scores = []
                    for col in detector_cols:
                        avg_scores.append(vis_df[col].mean())
                    
                    detector_df = pd.DataFrame({
                        'Detector': detector_names,
                        'Avg Score': avg_scores
                    })
                    
                    detector_df = detector_df.sort_values('Avg Score', ascending=False)
                    sns.barplot(x='Avg Score', y='Detector', data=detector_df, ax=ax, palette='viridis')
                    ax.set_title('Сравнение детекторов по средней оценке аномалий')
                    ax.set_xlabel('Средняя оценка')
                    ax.set_ylabel('Детектор')
            except Exception as e:
                self.handle_plot_error(ax, 'Невозможно построить сравнение детекторов', e)
        else:
            self.handle_plot_error(ax, 'Нет данных от разных детекторов для сравнения')


class SequenceVisualizer(BaseVisualizer):
    """
    Визуализатор анализа последовательностей.
    
    Позволяет создавать визуализации результатов работы детектора последовательностей,
    включая графы переходов, тепловые карты вероятностей и аномальные последовательности.
    """
    
    def __init__(self, max_sequences: int = 10, **kwargs):
        """Инициализация визуализатора последовательностей."""
        super().__init__(**kwargs)
        self.max_sequences = max_sequences
    
    def visualize_sequence_analysis(self, sequence_detector, data: pd.DataFrame, 
                                  show: bool = True, save: bool = False, 
                                  filename: Optional[str] = None) -> plt.Figure:
        """Визуализация анализа последовательностей."""
        if not hasattr(sequence_detector, 'transition_probs') or not sequence_detector.transition_probs:
            print("Детектор последовательностей не обучен или не содержит информации о переходах.")
            fig, ax = self.setup_figure("Анализ последовательностей")
            self.handle_plot_error(ax, "Детектор не содержит информации о переходах.")
            return fig
        
        # Предобработка данных
        preprocessed_data = sequence_detector.preprocess_data(data)
        states = preprocessed_data['state'].tolist()
        predictions = sequence_detector.predict(data)
        anomaly_indices = predictions[predictions['predicted_anomaly'] == 1].index
        
        if len(anomaly_indices) == 0:
            print("Аномальные последовательности не обнаружены.")
            fig, ax = self.setup_figure("Анализ последовательностей")
            self.handle_plot_error(ax, "Аномальные последовательности не обнаружены.")
            return fig
        
        # Сортируем аномалии по убыванию оценки аномальности
        sorted_anomalies = predictions.loc[anomaly_indices].sort_values('anomaly_score', ascending=False)
        top_anomalies = sorted_anomalies.head(self.max_sequences)
        
        # Создаем фигуру
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # 1. Визуализация графа переходов
        ax1 = plt.subplot(2, 2, 1)
        self._plot_transition_graph(ax1, sequence_detector)
        
        # 2. Тепловая карта вероятностей переходов
        ax2 = plt.subplot(2, 2, 2)
        self._plot_transition_heatmap(ax2, sequence_detector)
        
        # 3. Распределение аномальных оценок
        ax3 = plt.subplot(2, 2, 3)
        self._plot_anomaly_score_distribution(ax3, predictions)
        
        # 4. Топ аномальных последовательностей
        ax4 = plt.subplot(2, 2, 4)
        self._plot_anomaly_sequences(ax4, top_anomalies, states, sequence_detector)
        
        # Подгоняем лейауты
        plt.tight_layout()
        
        # Отображаем график
        if show:
            plt.show()
            
        # Выводим статистику
        self._print_sequence_statistics(sequence_detector, predictions, anomaly_indices)
        
        # Сохраняем график, если нужно
        if save:
            self.save_figure(fig, filename)
            
        return fig
    
    def _plot_transition_graph(self, ax: plt.Axes, sequence_detector) -> None:
        """Построение графа переходов между состояниями."""
        try:
            import networkx as nx
            
            # Создаем граф переходов
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
                             width=edge_weights, arrowsize=10, ax=ax)
            
            # Добавляем метки только для ключевых узлов
            if len(G.nodes()) <= 10:
                nx.draw_networkx_labels(G, pos)
            else:
                top_nodes = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
                nodes_to_label = {node: node for node, _ in top_nodes}
                nx.draw_networkx_labels(G, pos, labels=nodes_to_label)
            
            ax.set_title('Граф переходов между состояниями')
            ax.axis('off')
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить граф переходов', e)
    
    def _plot_transition_heatmap(self, ax: plt.Axes, sequence_detector) -> None:
        """Построение тепловой карты вероятностей переходов."""
        try:
            # Получаем топ-N состояний для тепловой карты
            all_states = set()
            for state in sequence_detector.transition_probs:
                all_states.add(state)
                all_states.update(sequence_detector.transition_probs[state].keys())
            
            # Выбираем наиболее важные состояния
            if len(all_states) <= 15:
                important_states = list(all_states)
            else:
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
                       xticklabels=important_states, yticklabels=important_states, ax=ax)
            ax.set_title('Тепловая карта вероятностей переходов')
            ax.set_xlabel('Следующее состояние')
            ax.set_ylabel('Текущее состояние')
            
            # Скрываем метки состояний, если их слишком много
            if len(important_states) > 8:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить тепловую карту переходов', e)
    
    def _plot_anomaly_score_distribution(self, ax: plt.Axes, predictions: pd.DataFrame) -> None:
        """Построение распределения аномальных оценок."""
        try:
            # Отображаем гистограмму аномальных оценок
            sns.histplot(data=predictions, x='anomaly_score', hue='predicted_anomaly',
                        kde=True, palette={0: 'blue', 1: 'red'}, ax=ax)
            ax.set_title('Распределение аномальных оценок')
            ax.set_xlabel('Аномальная оценка')
            ax.set_ylabel('Количество')
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить распределение аномальных оценок', e)
    
    def _plot_anomaly_sequences(self, ax: plt.Axes, top_anomalies: pd.DataFrame, states: List, sequence_detector) -> None:
        """Построение таблицы топ аномальных последовательностей."""
        try:
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
                
                sequence_str = ' -> '.join(map(str, sequence))
                
                # Если строка слишком длинная, сокращаем
                if len(sequence_str) > 30:
                    sequence_str = sequence_str[:27] + '...'
                
                cell_text.append([i+1, sequence_str, f"{row['anomaly_score']:.4f}"])
            
            # Если нет аномальных последовательностей, показываем сообщение
            if not cell_text:
                self.handle_plot_error(ax, 'Нет данных о аномальных последовательностях')
            else:
                # Создаем таблицу
                table = ax.table(
                    cellText=cell_text,
                    colLabels=['#', 'Последовательность', 'Оценка аномальности'],
                    loc='center',
                    cellLoc='center'
                )
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 1.5)
                ax.axis('off')
                ax.set_title('Топ аномальных последовательностей')
        except Exception as e:
            self.handle_plot_error(ax, 'Невозможно построить таблицу аномальных последовательностей', e)
    
    def _print_sequence_statistics(self, sequence_detector, predictions: pd.DataFrame, anomaly_indices) -> None:
        """Вывод статистики по анализу последовательностей."""
        print(f"Всего уникальных состояний: {len(sequence_detector.frequency_dict)}")
        print(f"Всего переходов между состояниями: {sum(len(transitions) for transitions in sequence_detector.transition_probs.values())}")
        print(f"Обнаружено аномальных последовательностей: {len(anomaly_indices)} ({len(anomaly_indices)/len(predictions)*100:.2f}%)")


# Функции-обертки для обратной совместимости
def visualize_ensemble_results(result_df, max_samples=1000, figsize=(20, 18)):
    """
    Усовершенствованная визуализация результатов ансамблевого детектора аномалий.
    
    Обертка для обратной совместимости со старым API.
    """
    visualizer = EnsembleVisualizer(max_samples=max_samples, figsize=figsize)
    visualizer.visualize_ensemble_results(result_df)


def visualize_sequence_analysis(sequence_detector, data, max_sequences=10, figsize=(15, 10)):
    """
    Визуализация анализа последовательностей для детектора SequenceAnomalyDetector.
    
    Обертка для обратной совместимости со старым API.
    """
    visualizer = SequenceVisualizer(max_sequences=max_sequences, figsize=figsize)
    visualizer.visualize_sequence_analysis(sequence_detector, data)
