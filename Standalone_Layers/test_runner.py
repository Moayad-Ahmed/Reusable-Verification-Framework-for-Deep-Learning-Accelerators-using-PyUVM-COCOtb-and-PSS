import os
from pathlib import Path
import shutil
from cocotb_tools.runner import get_runner

def clean_previous_run(root_dir):
    """
    Clean build directories and simulator artifacts.
    """
    print(f"Cleaning artifacts in {root_dir}...")

    # 1. Remove Directories
    dirs_to_remove = ["__pycache__", "sim_build", ".pytest_cache"]
    for d in dirs_to_remove:
        path = root_dir / d
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    # 2. Remove Specific Files
    files_to_remove = ["modelsim.ini", "transcript"]
    for f in files_to_remove:
        path = root_dir / f
        if path.exists():
            path.unlink()

    # 3. Remove File Patterns
    patterns_to_remove = ["*.ucdb", "*.xml", "*.yml", "*.vstf", "*.vcd"]
    for pattern in patterns_to_remove:
        for path in root_dir.glob(pattern):
            path.unlink()
            
    print("Cleanup completed successfully!")

def test_cnn_runner():
    # Run Cleanup
    root_dir = Path(__file__).resolve().parent
    clean_previous_run(root_dir)

    # QuestaSim installation path
    questa_path = os.getenv("QUESTA_HOME")

    if not questa_path:
        raise RuntimeError(
            "QUESTA_HOME is not set. Please set it to your Questa installation directory."
        )
    
    # Add it to the START of the system path so it is found first
    os.environ["PATH"] = questa_path + os.pathsep + os.environ["PATH"]

    rtl_dir = root_dir / "rtl"
    verilog_sources = [
        rtl_dir / "convolution_layer_generic.v",
        rtl_dir / "pooling_layer_generic.v",
        rtl_dir / "fully_connected_layer.v",
        rtl_dir / "Activation_layer_DUT.v",
        rtl_dir / "CNN_top.v",
    ]

    sim = os.getenv("SIM", "questa")
    runner = get_runner(sim)

    runner.build(
        sources=verilog_sources,
        hdl_toplevel="CNN_top",
        always=True,  # Forces a rebuild every time
    )

    runner.test(
        hdl_toplevel="CNN_top",
        test_module="pyuvm_components.test", # Your test module path
        waves=True,                          # Handles dumping waveforms automatically
    )

if __name__ == "__main__":
    test_cnn_runner()