import os
from pathlib import Path
import shutil
from cocotb_tools.runner import get_runner
from cocotb_tools.runner import Verilog


def clean_previous_run(root_dir):
    """
    Clean build directories and simulator artifacts.
    """
    print(f"Cleaning artifacts in {root_dir}...")

    # 1. Remove Directories
    dirs_to_remove = [
        "__pycache__",
        "sim_build",
        ".pytest_cache",
        "pyuvm_components/__pycache__",
        "strategy/__pycache__",
        "utils/__pycache__"
    ]
    for d in dirs_to_remove:
        path = root_dir / d
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)

    # 2. Remove Specific Files
    files_to_remove = [
        "modelsim.ini",
        "transcript",
        "cdma2cbuf_data_rtl.dat",
        "cdma2cbuf_weight_rtl.dat"
    ]
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


def parse_rtl_sources_file(root_dir):
    """
    Parse the rtl_sources.f file and return list of absolute source file paths.
    Includes .vlib files as they contain important module definitions.
    Tags .vlib files explicitly as Verilog since cocotb doesn't recognize the extension.
    """
    rtl_sources_file = root_dir / "rtl_sources.f"
    sources = []
    
    if not rtl_sources_file.exists():
        raise FileNotFoundError(f"rtl_sources.f not found at {rtl_sources_file}")
    
    with open(rtl_sources_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#') and not line.startswith('//'):
                # Convert relative path to absolute
                source_path = root_dir / line
                if source_path.exists():
                    # Tag .vlib files as Verilog since cocotb doesn't recognize them
                    if source_path.suffix == '.vlib':
                        sources.append(Verilog(source_path))
                    else:
                        sources.append(str(source_path))
                else:
                    print(f"Warning: Source file not found: {source_path}")
    
    return sources


def test_nvdla_runner():
    """
    Main test runner for NVDLA framework.
    """
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

    # Parse RTL sources from rtl_sources.f (includes .vlib files)
    print("Parsing RTL sources from rtl_sources.f...")
    rtl_sources = parse_rtl_sources_file(root_dir)
    print(f"Found {len(rtl_sources)} RTL source files")

    # Add the top-level files
    rtl_dir = root_dir / "rtl"
    verilog_sources = rtl_sources + [
        str(rtl_dir / "dram.sv"),
        str(rtl_dir / "NVDLA_top.sv"),
    ]

    # Setup include directories
    include_dirs = [
        root_dir / "rtl",
        root_dir / "rtl" / "vmod" / "include",
        root_dir / "rtl" / "vmod" / "vlibs",
        root_dir / "rtl" / "vmod" / "rams" / "model",
        root_dir / "rtl" / "vmod" / "rams" / "synth",
        root_dir / "rtl" / "vmod" / "nvdla",
        root_dir / "rtl" / "spec" / "odif",
        root_dir / "rtl" / "spec" / "manual",
        root_dir / "rtl" / "vmod" / "defs",
    ]

    # Add Python source directories to PYTHONPATH
    python_paths = [
        str(root_dir / "pyuvm_components"),
        str(root_dir / "strategy"),
        str(root_dir / "utils"),
    ]
    
    # Add to PYTHONPATH using semicolon for Windows
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = ";".join(python_paths) + ";" + os.environ["PYTHONPATH"]
    else:
        os.environ["PYTHONPATH"] = ";".join(python_paths)

    print(f"PYTHONPATH set to: {os.environ['PYTHONPATH']}")

    # Get simulator
    sim = os.getenv("SIM", "questa")
    runner = get_runner(sim)

    # Build parameters
    build_args = []
    if sim == "questa":
        build_args.append("+define+SYNTHESIS")
    
    print("Building NVDLA design...")
    runner.build(
        sources=verilog_sources,
        hdl_toplevel="NVDLA_top",
        includes=include_dirs,
        build_args=build_args,
        always=True,  # Forces a rebuild every time
    )

    print("Running NVDLA tests...")
    runner.test(
        hdl_toplevel="NVDLA_top",
        test_module="pyuvm_components.test",  # Your test module path
        hdl_toplevel_lang="verilog",
        waves=True,  # Handles dumping waveforms automatically
    )

    print("Test execution completed!")


if __name__ == "__main__":
    test_nvdla_runner()
