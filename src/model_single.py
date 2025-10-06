import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional
import numpy as np

class ComplexModel(nn.Module):
    def __init__(self, params: List[Tuple[Tensor, ...]]):
        super().__init__()
        if len(params) < 2:
            raise ValueError("Params must contain at least global weights and one model")
        
        self.models = nn.ModuleList([CustomModel(*p) for p in params[1:]])
        self.r_global = nn.Parameter(params[0], requires_grad=True)
        
    def forward(self, target_Hz: Tensor = torch.tensor([0.0])) -> Tensor:
        individual_spectra = []
        for model in self.models:
            spectrum = model(target_Hz)
            individual_spectra.append(spectrum[:, 1])
        
        weights = F.softmax(self.r_global, dim=0).reshape(-1, 1)
        combined_intensity = torch.stack(individual_spectra) * weights
        combined_intensity = combined_intensity.sum(dim=0)
        
        frequencies = self.models[0](target_Hz)[:, 0]
        return torch.stack((frequencies, combined_intensity), -1)

class CustomModel(nn.Module):
    def __init__(
        self,
        v: Tensor,
        J: Tensor,
        w: Tensor,
        r: Tensor,
        protons_number: Tensor,
        freq: Tensor,
        ppm0: Tensor
    ):
        super().__init__()
        
        self._validate_inputs(v, J, w, r, protons_number, freq, ppm0)
        
        self.v0 = nn.Parameter((ppm0 * freq).double(), requires_grad=True)
        self.J0 = nn.Parameter(J.double(), requires_grad=True)
        self.w0 = nn.Parameter(w.double(), requires_grad=True)
        self.r0 = nn.Parameter(r.double(), requires_grad=True)
        self.proton_config = nn.Parameter(protons_number.double(), requires_grad=False)
        self.freq = nn.Parameter(freq, requires_grad=False)
        self.ppm0 = nn.Parameter(ppm0.double(), requires_grad=False)
        
        self.PPrint = False
        self.nuclei_count = self._calculate_nuclei_count(protons_number)
        self.proton_groups = self._process_proton_groups(protons_number)
        
        self.initialize_projection_matrices()
        
    def _validate_inputs(
        self,
        v: Tensor,
        J: Tensor,
        w: Tensor,
        r: Tensor,
        protons_number: Tensor,
        freq: Tensor,
        ppm0: Tensor
    ) -> None:
        if J.dim() != 2 or J.shape[0] != J.shape[1]:
            raise ValueError("J must be a square matrix")
        if v.shape[0] != J.shape[0]:
            raise ValueError("v must have same dimension as J matrix")
        if protons_number.sum() <= 0:
            raise ValueError("Total proton count must be positive")

    def _calculate_nuclei_count(self, protons_number: Tensor) -> int:
        return protons_number.sum().long().item()

    def _process_proton_groups(self, protons_number: Tensor) -> Tensor:
        return protons_number if protons_number.shape[0] > 1 else torch.tensor([1.0])

    def initialize_projection_matrices(self) -> None:
        num_groups = self.proton_groups.shape[0]
        total_spins = self.nuclei_count
        
        self.mat1 = torch.zeros((num_groups, total_spins), dtype=torch.double)
        cumulative_sum = torch.cumsum(self.proton_groups, dim=0).long()
        
        start_idx = 0
        for i, end_idx in enumerate(cumulative_sum):
            self.mat1[i, start_idx:end_idx] = 1
            start_idx = end_idx
            
        self.mat2 = self.mat1.T

    def compute_spin_operators(self) -> Tuple[Tensor, Tensor]:
        nspins = self.nuclei_count
        
        sigma_x = torch.tensor([[0, 1 / 2], [1 / 2, 0]], dtype=torch.float64)
        real_part = torch.tensor([[0, 0], [0, 0]], dtype=torch.float64)
        imag_part = torch.tensor([[0, -1 / 2], [1 / 2, 0]], dtype=torch.float64)
        sigma_y = torch.complex(real_part, imag_part)
        sigma_z = torch.tensor([[1 / 2, 0], [0, -1 / 2]], dtype=torch.float64)
        identity = torch.tensor([[1, 0], [0, 1]], dtype=torch.float64)
        
        spin_operators = torch.empty((3, nspins, 2 ** nspins, 2 ** nspins), dtype=torch.complex128)
        
        for spin_idx in range(nspins):
            Lx = torch.tensor([1.0], dtype=torch.complex128)
            Ly = torch.tensor([1.0], dtype=torch.complex128)
            Lz = torch.tensor([1.0], dtype=torch.complex128)

            for k in range(nspins):
                if k == spin_idx:
                    Lx = torch.kron(Lx, sigma_x)
                    Ly = torch.kron(Ly, sigma_y)
                    Lz = torch.kron(Lz, sigma_z)
                else:
                    Lx = torch.kron(Lx, identity)
                    Ly = torch.kron(Ly, identity)
                    Lz = torch.kron(Lz, identity)

            spin_operators[0, spin_idx] = Lx
            spin_operators[1, spin_idx] = Ly
            spin_operators[2, spin_idx] = Lz

        operators_transposed = torch.transpose(spin_operators, 1, 0)
        product_operators = torch.tensordot(operators_transposed, spin_operators, dims=([1, 3], [0, 2])).swapaxes(1, 2)
        
        return spin_operators[2], product_operators

    def compute_transition_matrices(self) -> Tuple[Tensor, Tensor]:
        num_states = 2 ** self.nuclei_count
        T1 = torch.zeros((num_states, num_states), dtype=torch.double)
        T2 = torch.zeros((num_states, num_states), dtype=torch.double)
        
        for i in range(num_states - 1):
            for j in range(i + 1, num_states):
                if bin(i ^ j).count('1') == 1:
                    spin_idx = int(self.nuclei_count - 1 - torch.log2(torch.tensor(j - i)))
                    
                    relaxation_param = torch.mm(torch.abs(self.r0).reshape(1, -1), self.mat1).squeeze(0)[spin_idx]
                    width_param = torch.mm(torch.abs(self.w0).reshape(1, -1), self.mat1).squeeze(0)[spin_idx]
                    
                    T1[i, j] = torch.sqrt(relaxation_param)
                    T2[i, j] = torch.sqrt(width_param * relaxation_param)
        
        T1_symmetric = T1 + T1.T - 2 * torch.diag(torch.diag(T1))
        T2_symmetric = T2 + T2.T - 2 * torch.diag(torch.diag(T2))
        
        return T1_symmetric, T2_symmetric

    def compute_hamiltonian(self) -> Tensor:
        Lz = self.Lz.double()

        chemical_shift_contribution = torch.mm(self.v0.reshape(1, -1), self.mat1).squeeze(0).double()
        H_chemical = torch.tensordot(chemical_shift_contribution, Lz, dims=1)
        H_chemical = H_chemical.to(torch.complex128)

        J_processed = torch.mm(self.mat2, torch.mm(self.J0, self.mat1))
        coupling_scalars = 0.5 * J_processed.to(torch.complex128)
        H_coupling = torch.tensordot(coupling_scalars, self.Lproduct, dims=2)

        return H_chemical + H_coupling

    def compute_spectral_peaks(self, intensity_cutoff: float = 0.001) -> Tensor:
        hamiltonian = self.compute_hamiltonian()
        
        with torch.no_grad():
            energy_levels, eigenstates = torch.linalg.eig(hamiltonian)
        
        if self.v0.requires_grad:
            eigenvalues = torch.linalg.eigvals(hamiltonian)
        else:
            eigenvalues = torch.linalg.eigvals(hamiltonian.requires_grad_(True)).detach()
        
        # Исправленная обработка собственных векторов
        processed_eigenstates = []
        for element in eigenstates.reshape(-1):
            element_real = element.real
            if torch.isinf(element_real) or torch.isnan(element_real) or element_real == 0:
                processed_eigenstates.append(torch.tensor(0.0))
            else:
                processed_eigenstates.append(element_real)
        
        # Убедимся, что все тензоры имеют одинаковый размер
        eigenvector_matrix = torch.stack(processed_eigenstates).reshape(2**self.nuclei_count, 2**self.nuclei_count)
        
        intensities = torch.square(torch.mm(eigenvector_matrix.T, torch.mm(self.T1, eigenvector_matrix)))
        widths = torch.square(torch.mm(eigenvector_matrix.T, torch.mm(self.T2, eigenvector_matrix)))
        
        peak_data = self.compile_peak_data(intensities, eigenvalues.real, widths, intensity_cutoff)
        normalized_peaks = self.normalize_peak_intensities(peak_data)
        
        return normalized_peaks

    def compile_peak_data(self, intensities: Tensor, energies: Tensor, widths: Tensor, cutoff: float) -> Tensor:
        upper_intensities = torch.triu(intensities)
        upper_widths = torch.triu(widths)
        energy_differences = torch.abs(energies[:, None] - energies)
        upper_energies = torch.triu(energy_differences)
        
        combined_data = torch.stack([upper_energies, upper_intensities, upper_widths])
        flattened_data = combined_data.reshape(3, intensities.shape[0] ** 2).T
        filtered_data = flattened_data[flattened_data[:, 1] >= cutoff]
        
        width_ratios = filtered_data[:, 2] / (filtered_data[:, 1] + 1e-10)
        return torch.hstack((filtered_data[:, :1], filtered_data[:, 1:2], width_ratios.reshape(-1, 1)))

    def normalize_peak_intensities(self, peak_data: Tensor) -> Tensor:
        frequencies, intensities, widths = peak_data[:, 0], peak_data[:, 1], peak_data[:, 2]
        normalized_intensities = self.apply_intensity_normalization(intensities)
        return torch.stack((frequencies, normalized_intensities, widths), dim=1)

    def apply_intensity_normalization(self, intensities: Tensor, spin_count: int = 1) -> Tensor:
        normalization_factor = spin_count / (torch.sum(intensities) + 1e-10)
        return normalization_factor * intensities

    def generate_lorentzian_spectrum(self, frequency_range: Tensor, peak_data: Tensor) -> Tensor:
        spectrum = torch.zeros_like(frequency_range)
        
        for peak in peak_data:
            center_frequency, intensity, width = peak
            spectrum += self.single_lorentzian_peak(frequency_range, center_frequency, intensity, width)
        
        return spectrum

    def single_lorentzian_peak(self, frequencies: Tensor, center: float, intensity: float, width: float) -> Tensor:
        scaling_factor = 0.5 / (width + 1e-10)
        denominator = (0.5 * width) ** 2 + (frequencies - center) ** 2 + 1e-10
        return scaling_factor * intensity * ((0.5 * width) ** 2) / denominator

    def compute_trapezoidal_integral(self, y_values: Tensor, x_values: Tensor) -> Tensor:
        x_differences = x_values[1:] - x_values[:-1]
        y_averages = (y_values[1:] + y_values[:-1]) / 2
        area_elements = x_differences * y_averages
        return torch.sum(area_elements)

    def get_frequency_limits(self) -> Tuple[float, float]:
        return 550.0, 2500.0

    def forward(self, target_Hz: Tensor = torch.tensor([0.0])) -> Tensor:
        self.initialize_projection_matrices()
        self.T1, self.T2 = self.compute_transition_matrices()
        self.Lz, self.Lproduct = self.compute_spin_operators()
        
        peak_data = self.compute_spectral_peaks()
        
        sorted_indices = torch.argsort(peak_data[:, 0], descending=True)
        sorted_peaks = peak_data[sorted_indices]
        
        
        
        if target_Hz.shape[0] > 1:
            frequency_points = target_Hz.float()
            spectrum_values = self.generate_lorentzian_spectrum(frequency_points, sorted_peaks)
            spectrum_integral = self.compute_trapezoidal_integral(spectrum_values, frequency_points)
        else:
            num_points = 12758
            lower_limit, upper_limit = self.get_frequency_limits()
            frequency_points = torch.linspace(lower_limit, upper_limit, num_points)
            spectrum_values = self.generate_lorentzian_spectrum(frequency_points, sorted_peaks)
            spectrum_integral = self.compute_trapezoidal_integral(spectrum_values, frequency_points)
        
        normalized_spectrum = spectrum_values / (spectrum_integral + 1e-10)
        
        return torch.stack((frequency_points, normalized_spectrum), -1)