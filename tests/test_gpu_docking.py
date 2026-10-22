import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
from physics_engine.dock_vina_gpu import execute_gpu_batch, parse_gpu_results

def test_gpu_batch_execution_success(tmp_path: Path):
    target_config = {"disease": "Test", "pdbqt": "rec.pdbqt", "grid": "grid.json"}
    
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        mock_process.communicate.return_value = ("stdout", "")
        mock_process.poll.return_value = 0  # Process finished successfully
        mock_process.returncode = 0
        
        assert execute_gpu_batch("Vina-GPU", str(tmp_path), target_config, str(tmp_path)) is True
        mock_process.kill.assert_not_called()

def test_gpu_batch_execution_timeout(tmp_path: Path):
    target_config = {"disease": "Test", "pdbqt": "rec.pdbqt", "grid": "grid.json"}
    
    with patch("subprocess.Popen") as mock_popen:
        mock_process = MagicMock()
        mock_popen.return_value = mock_process
        
        mock_process.poll.return_value = None  # Process still running when timeout hits
        mock_process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd="Vina-GPU", timeout=60),
            ("", "") # Second call in finally block succeeds
        ]
        
        assert execute_gpu_batch("Vina-GPU", str(tmp_path), target_config, str(tmp_path), timeout=60) is False
        mock_process.kill.assert_called_once()

def test_gpu_result_parsing_strict_enforcement(tmp_path: Path):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    
    (output_dir / "MOL-1_out.pdbqt").write_text("REMARK VINA RESULT:    -8.5      0.000      0.000\n")
    
    # Missing count
    with pytest.raises(ValueError, match="Expected 2 results, found 1"):
        parse_gpu_results(str(output_dir), expected_count=2)
        
    (output_dir / "MOL-2_out.pdbqt").write_text("MODEL 1\nATOM 1") # Missing REMARK line
    
    # Missing result line
    with pytest.raises(ValueError, match="Missing or malformed VINA RESULT in MOL-2_out.pdbqt"):
        parse_gpu_results(str(output_dir), expected_count=2)