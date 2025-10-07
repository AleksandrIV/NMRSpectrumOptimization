import torch
import torch.nn as nn
from torch import Tensor
from pathlib import Path
import numpy as np
from model_single import ComplexModel
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional, Any
import logging
from IPython.display import display, Math


# Настройка логирования
logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Кастомное исключение для ошибок загрузки моделей"""
    pass



class SpectrumVisualizationError(Exception):
    """Кастомное исключение для ошибок визуализации спектров."""
    pass



class StateDictGenerator:
    """
    Генератор state_dict для комплексных моделей.
    
    Attributes:
        num_models (int): Количество моделей
        state_dict (OrderedDict): Словарь параметров моделей
    """
    
    # Константы для имен параметров
    PARAM_NAMES = ['v0', 'J0', 'w0', 'r0', 'proton_config', 'freq', 'ppm0']
    GLOBAL_PARAM = 'r_global'
    MODEL_PREFIX = 'models.'
    
    def __init__(self, num_models: int):
        """
        Args:
            num_models: Количество моделей для инициализации
        """
        if num_models <= 0:
            raise ValueError("num_models must be positive")
            
        self.num_models = num_models
        self.state_dict = OrderedDict()
        self.state_dict[self.GLOBAL_PARAM] = torch.ones(num_models)

    def set_param(self, model_idx: int, param_name: str, value: torch.Tensor) -> None:
        """Устанавливает параметр для конкретной модели.
        
        Args:
            model_idx: Индекс модели (0-based)
            param_name: Имя параметра из PARAM_NAMES
            value: Значение параметра как torch.Tensor
            
        Raises:
            ValueError: Если model_idx или param_name некорректны
        """
        if not (0 <= model_idx < self.num_models):
            raise ValueError(f"model_idx must be between 0 and {self.num_models - 1}")
            
        if param_name not in self.PARAM_NAMES:
            raise ValueError(f"param_name must be one of {self.PARAM_NAMES}")

        if param_name == 'ppm0':
            self.state_dict[f"{self.MODEL_PREFIX}{model_idx}.{'v0'}"] = value*self.state_dict[f"{self.MODEL_PREFIX}{model_idx}.{'freq'}"]   
            
        self.state_dict[f"{self.MODEL_PREFIX}{model_idx}.{param_name}"] = value

    def get_state_dict(self) -> OrderedDict:
        """Возвращает полный state_dict.
        
        Returns:
            OrderedDict: State dictionary готовый для сохранения
        """
        return self.state_dict

    def get_pth(self) -> List[Any]:
        """Конвертирует state_dict в формат PTH.
        
        Returns:
            List: Список параметров в формате PTH
            
        Raises:
            ModelLoadError: Если state_dict содержит некорректные данные
        """
        try:
            return load_pth_from_state_dict(self.state_dict)
        except Exception as e:
            raise ModelLoadError(f"Failed to convert state_dict to PTH: {e}")


def _extract_model_indices(state_dict: Dict[str, Any]) -> List[int]:
    """Извлекает индексы моделей из state_dict.
    
    Args:
        state_dict: Словарь параметров модели
        
    Returns:
        List[int]: Отсортированный список индексов моделей
    """
    model_indices = []
    for key in state_dict.keys():
        if key.startswith(StateDictGenerator.MODEL_PREFIX):
            try:
                # Извлекаем индекс модели: "models.0.v0" -> 0
                model_idx = int(key.split('.')[1])
                if model_idx not in model_indices:
                    model_indices.append(model_idx)
            except (IndexError, ValueError):
                logger.warning(f"Skipping invalid model key: {key}")
                continue
                
    return sorted(model_indices)


def _load_model_params(state_dict: Dict[str, Any], model_idx: int) -> Tuple:
    """Загружает параметры для конкретной модели.
    
    Args:
        state_dict: Словарь параметров модели
        model_idx: Индекс модели
        
    Returns:
        Tuple: Кортеж параметров модели
        
    Raises:
        KeyError: Если отсутствуют обязательные параметры
    """
    prefix = f"{StateDictGenerator.MODEL_PREFIX}{model_idx}."
    
    # Проверяем наличие всех обязательных параметров
    required_params = StateDictGenerator.PARAM_NAMES
    missing_params = [param for param in required_params 
                     if f"{prefix}{param}" not in state_dict]
    
    if missing_params:
        raise KeyError(f"Missing parameters for model {model_idx}: {missing_params}")
    
    # Загружаем и конвертируем параметры
    v0 = state_dict[f"{prefix}v0"].double()
    J0 = state_dict[f"{prefix}J0"].double()
    w0 = state_dict[f"{prefix}w0"].double()
    r0 = state_dict[f"{prefix}r0"].double()
    proton_config = state_dict[f"{prefix}proton_config"].double()
    freq = state_dict[f"{prefix}freq"]  # Частота может быть любого типа
    ppm0 = state_dict[f"{prefix}ppm0"].double()
    
    return (v0, J0, w0, r0, proton_config, freq, ppm0)


def load_pth_from_state_dict(state_dict: Dict[str, Any]) -> List[Any]:
    """Загружает параметры из state_dict в формат PTH.
    
    Args:
        state_dict: Словарь параметров модели
        
    Returns:
        List: Список параметров в формате PTH
        
    Raises:
        ModelLoadError: Если state_dict некорректен
    """
    try:
        # Извлекаем индексы моделей
        model_indices = _extract_model_indices(state_dict)
        
        if not model_indices:
            raise ModelLoadError("No valid models found in state_dict")
        
        # Загружаем параметры для каждой модели
        model_params = []
        for model_idx in model_indices:
            params = _load_model_params(state_dict, model_idx)
            model_params.append(params)
        
        # Обрабатываем глобальные параметры
        if StateDictGenerator.GLOBAL_PARAM in state_dict:
            pext = [state_dict[StateDictGenerator.GLOBAL_PARAM]]
        else:
            pext = [torch.ones(len(model_indices))]
            
        pext.extend(model_params)
        
        return pext
        
    except Exception as e:
        raise ModelLoadError(f"Failed to load PTH from state_dict: {e}")


def load_pth_file(filename: str) -> List[Any]:
    """Загружает параметры из файла .pth.
    
    Args:
        filename: Путь к файлу .pth
        
    Returns:
        List: Список параметров в формате PTH
        
    Raises:
        ModelLoadError: Если файл не может быть загружен
    """
    try:
        state_dict = torch.load(filename, map_location='cpu')
        logger.info(f"Successfully loaded state_dict from {filename}")
        return load_pth_from_state_dict(state_dict)
    except Exception as e:
        raise ModelLoadError(f"Failed to load PTH file {filename}: {e}")


def load_pth_model(filename: str) -> ComplexModel:
    """Загружает и создает ComplexModel из файла .pth.
    
    Args:
        filename: Путь к файлу .pth
        
    Returns:
        ComplexModel: Загруженная модель
        
    Raises:
        ModelLoadError: Если модель не может быть создана
    """
    try:
        pth_params = load_pth_file(filename)
        model = ComplexModel(pth_params)
        logger.info(f"Successfully created ComplexModel from {filename}")
        return model
    except Exception as e:
        raise ModelLoadError(f"Failed to create model from {filename}: {e}")

def code_matrix(array: Tensor, key: str) -> str:
    """
    Convert tensor to LaTeX matrix representation.
    
    Args:
        array: Input tensor
        key: Matrix name for LaTeX
        
    Returns:
        LaTeX string representation of matrix
    """
    if not isinstance(array, Tensor):
        raise TypeError("array must be a torch.Tensor")
    
    # Ensure 2D shape
    if len(array.shape) < 2:
        array = array.reshape(1, -1)
    
    matrix_content = ''
    for row in array:
        if len(row.shape) == 0:  # Scalar case
            matrix_content += f"{row.item():.4f}\\\\"
        else:
            row_content = " & ".join(f"{number.item():.4f}" for number in row)
            matrix_content += f"{row_content}\\\\"
    
    matrix_content = matrix_content[:-2]  # Remove last \\
    
    return f"{key} = " + r"\begin{bmatrix}" + matrix_content + r"\end{bmatrix}"        
        
    
    
def save_optimization_results(
    model: nn.Module,
    initial_metric: Tensor,
    final_metric: Tensor,
    metabolite: str,
    output_dir: Path = Path(".")
) -> None:
    """
    Save optimization results to HTML file.
    
    Args:
        model: Optimized model
        initial_metric: Initial loss metric
        final_metric: Final loss metric
        metabolite: Metabolite name
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_file = output_dir / f"{metabolite}.html"
    
    try:
        # Read template HTML if exists
        if html_file.exists():
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            sections = html_content.rsplit('</div>', 1)
            original_content = sections[0] + '</div>' if len(sections) > 1 else ""
        else:
            original_content = ""
            sections = [""]
        
        # Generate results section
        results_section = _generate_results_html(model, initial_metric, final_metric)
        
        # Combine content
        final_html = original_content + results_section + (sections[1] if len(sections) > 1 else "")
        
        # Write output
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        logger.info(f"Results saved to {html_file}")
        
    except Exception as e:
        raise SpectrumVisualizationError(f"Failed to save results: {e}")


def _generate_results_html(model: nn.Module, initial_metric: Tensor, final_metric: Tensor) -> str:
    """
    Generate HTML section with optimization results.
    
    Args:
        model: Optimized model
        initial_metric: Initial metric
        final_metric: Final metric
        
    Returns:
        HTML string with results
    """
    mathjax_scripts = '''
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    '''
    
    results_html = mathjax_scripts
    
    # Add model parameters
    for param_name, param_tensor in model.state_dict().items():
        display_name = r'proton\:config' if param_name == 'proton_config' else param_name
        latex_formula = code_matrix(param_tensor, display_name)
        results_html += f'''
        <div style="font-size: 24px;">
            \\[ {latex_formula} \\]
        </div>
        '''
    
    # Add metrics
    metrics_latex = f"metric\\:before\\: = {initial_metric.item():.6f}, after = {final_metric.item():.6f}"
    results_html += f'''
    <div style="font-size: 24px;">
        \\[ {metrics_latex} \\]
    </div>
    '''
    
    return results_html


def display_spin_system(
    spin_system: OrderedDict,
    param_descriptions: Optional[Dict[str, str]] = None
) -> None:
    """
    Display spin system parameters from OrderedDict with LaTeX formatting.
    
    Args:
        spin_system: OrderedDict with model parameters
        param_descriptions: Dictionary with parameter descriptions
    """
    if param_descriptions is None:
        param_descriptions = {
            'r_global': r'Global\ R\ factor',
            'v0': r'Chemical\ shifts\ (Hz)',
            'J0': r'J\ couplings\ (Hz)',
            'w0': r'Widths',
            'r0': r'R\ factors',
            'proton_config': r'Proton\ configuration',
            'freq': r'Spectrometer\ frequency\ (MHz)',
            'ppm0': r'Chemical\ shifts\ (ppm)'
        }
    
    # Extract models and global parameters
    models = {}
    global_params = {}
    
    for key, tensor in spin_system.items():
        if key == 'r_global':
            global_params['r_global'] = tensor
        elif key.startswith('models.'):
            parts = key.split('.')
            if len(parts) >= 3:
                try:
                    model_idx = int(parts[1])
                    param_name = parts[2]
                    
                    if model_idx not in models:
                        models[model_idx] = {}
                    models[model_idx][param_name] = tensor
                except (ValueError, IndexError):
                    logger.warning(f"Skipping invalid key: {key}")
    
    # Display global parameters
    if global_params:
        display(Math(r"\mathbf{Global\ Parameters}"))
        for param_name, tensor in global_params.items():
            desc = param_descriptions.get(param_name, param_name)
            if tensor.dim() == 1:
                values = [f"{x:.4f}" for x in tensor.detach().numpy()]
                latex_formula = f"{desc} = " + r"\begin{bmatrix}" + " & ".join(values) + r"\end{bmatrix}"
                display(Math(latex_formula))
        print()
    
    # Display model parameters
    for model_idx in sorted(models.keys()):
        display(Math(fr"\mathbf{{Spin\ System\ {model_idx}}}"))
        model_params = models[model_idx]
        
        param_order = ['ppm0', 'J0', 'w0', 'r0', 'proton_config', 'v0', 'freq']
        
        for param_name in param_order:
            if param_name in model_params:
                tensor = model_params[param_name]
                desc = param_descriptions.get(param_name, param_name)
                _display_tensor_as_latex(tensor, desc)
        
        if model_idx < len(models) - 1:
            print()


def _display_tensor_as_latex(tensor: Tensor, description: str) -> None:
    """
    Display tensor as LaTeX formatted output.
    
    Args:
        tensor: Tensor to display
        description: Tensor description
    """
    if tensor.dim() == 0:  # Scalar
        latex_formula = f"{description} = {tensor.item():.4f}"
    elif tensor.dim() == 1:  # Vector
        values = [f"{x:.4f}" for x in tensor.detach().numpy()]
        if len(values) <= 8:
            latex_formula = f"{description} = " + r"\begin{bmatrix}" + " & ".join(values) + r"\end{bmatrix}"
        else:
            latex_formula = (f"{description} = " + r"\begin{bmatrix}" + 
                           " & ".join(values[:4]) + r" & \cdots & " + values[-1] + r"\end{bmatrix}")
    elif tensor.dim() == 2:  # Matrix
        values = tensor.detach().numpy()
        rows, cols = values.shape
        
        if rows <= 5 and cols <= 5:
            matrix_str = r"\\".join([" & ".join([f"{x:.4f}" for x in row]) for row in values])
            latex_formula = f"{description} = " + r"\begin{bmatrix}" + matrix_str + r"\end{bmatrix}"
        else:
            latex_formula = f"{description} = " + r"\text{{matrix\ }}" + f"{rows}\\times{cols}"
            display(Math(latex_formula))
            
            # Display submatrix
            small_matrix = values[:min(3, rows), :min(3, cols)]
            if small_matrix.size > 0:
                matrix_str = r"\\".join([" & ".join([f"{x:.4f}" for x in row]) for row in small_matrix])
                latex_formula = (f"{description}_{{[1:{len(small_matrix)},1:{len(small_matrix[0])}]}} = " +
                               r"\begin{bmatrix}" + matrix_str + r"\end{bmatrix}")
            else:
                return
    else:  # Higher dimensional tensor
        latex_formula = f"{description} = " + r"\text{{tensor\ }}" + f"{tuple(tensor.shape)}"
    
    display(Math(latex_formula))
