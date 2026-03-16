"""
Basic usage example for ifc-hydro library.

This example demonstrates how to use the ifc-hydro library to analyze hydraulic systems from a synthetic IFC model.
"""

import ifcopenshell as ifc
from ifc_hydro import Base, Topology, Pressure
from ifc_hydro.visualization import GraphPlotter
import sys
import os


def main():
    """
    Main function demonstrating usage of ifc-hydro library.
    """

    # Configure log file for this run
    log_dir_input = input("Enter log directory (leave blank for current directory): ").strip()
    if not log_dir_input:
        log_dir_input = '.\ifc_hydro\examples\demo'

    log_name_input = input("Enter log file name (leave blank for ifc-hydro.log): ").strip()
    Base.configure_log(Base, log_dir=log_dir_input, log_name=log_name_input)

    # Load IFC model
    ifc_file_path = input("Enter IFC file path (leave blank for 'demo-project.ifc'): ").strip()
    if not ifc_file_path:
        ifc_file_path = '.\ifc_hydro\examples\demo\demo-project.ifc'

    # Validate IFC file exists
    if not os.path.exists(ifc_file_path):
        error_msg = f"ERROR: IFC file not found at path: {ifc_file_path}"
        Base.append_log(Base, error_msg)
        print(error_msg)
        sys.exit(1)

    # Load IFC model with error handling
    try:
        model = ifc.open(ifc_file_path)
        Base.append_log(Base, f"> Successfully loaded IFC model from: {ifc_file_path}")
    except Exception as e:
        error_msg = f"ERROR: Failed to open IFC file: {str(e)}"
        Base.append_log(Base, error_msg)
        print(error_msg)
        sys.exit(1)

    # Initialize topology creator with the model and calculate all paths
    try:
        topology = Topology(model)
        test_path = topology.all_paths_finder()
    except ValueError as e:
        error_msg = f"ERROR: Topology creation failed: {str(e)}"
        Base.append_log(Base, error_msg)
        print(error_msg)
        Base.append_log(Base, "Program halted due to topology errors.")
        sys.exit(1)
    except Exception as e:
        error_msg = f"ERROR: Unexpected error during topology creation: {str(e)}"
        Base.append_log(Base, error_msg)
        print(error_msg)
        sys.exit(1)

    # Validate that paths were created
    if not test_path or len(test_path) == 0:
        error_msg = "ERROR: No paths were created. Cannot proceed with pressure calculations."
        Base.append_log(Base, error_msg)
        print(error_msg)
        sys.exit(1)

    # Visualize the hydraulic system topology
    visualize_input = input("Visualize hydraulic system topology? (y/n, leave blank for no): ").strip().lower()
    if visualize_input == 'y':
        try:
            plotter = GraphPlotter()
            plotter.from_topology_paths(test_path)
            plotter.print_statistics()

            # Ask for visualization save path
            save_path_input = input("Enter path to save visualization (leave blank to display only): ").strip()
            save_path = save_path_input if save_path_input else None

            # Plot the paths
            plotter.plot_paths(
                paths=test_path,
                layout='hierarchical',
                title='Demo Project - Hydraulic System Topology',
                save_path=save_path,
                show=True
            )
            Base.append_log(Base, "> Successfully generated hydraulic system visualization")
        except Exception as e:
            error_msg = f"WARNING: Failed to generate visualization: {str(e)}"
            Base.append_log(Base, error_msg)
            print(error_msg)
            # Continue with pressure calculations despite visualization failure

    # Initialize pressure calculator with the model
    pressure_calc = Pressure(model)

    # Example: Calculate available pressure at a specific terminal
    # Get terminal ID from user or use default
    terminal_id_input = input("Enter terminal ID (leave blank to calculate all terminals): ").strip()
    if terminal_id_input:

        # Default terminal IDs from the 'projeto-demonstracao.ifc' model:
        # Shower         --> 5423
        # Wash and Basin --> 6986
        # WC Seat        --> 7061
        try:
            term_test = model.by_id(int(terminal_id_input))
            press_test = pressure_calc.available(term_test, test_path)
        except RuntimeError:
            error_msg = f"ERROR: Terminal with ID {terminal_id_input} not found in the IFC model."
            Base.append_log(Base, error_msg)
            print(error_msg)
            sys.exit(1)
        except Exception as e:
            error_msg = f"ERROR: Failed to calculate pressure: {str(e)}"
            Base.append_log(Base, error_msg)
            print(error_msg)
            sys.exit(1)

    else:

        terminals = model.by_type("IfcSanitaryTerminal")

        if not terminals:
            error_msg = "ERROR: No IfcSanitaryTerminal elements found in the IFC model."
            Base.append_log(Base, error_msg)
            print(error_msg)
            sys.exit(1)

        for terminal in terminals:
            try:
                # Get step numerical identifier for the terminal
                terminal_id = terminal.id()

                # Calculate available pressure at the terminal
                term_test = model.by_id(terminal_id)
                press_test = pressure_calc.available(term_test, test_path)
            except Exception as e:
                error_msg = f"ERROR: Failed to calculate pressure for terminal {terminal_id}: {str(e)}"
                Base.append_log(Base, error_msg)
                print(error_msg)
                # Continue with next terminal instead of exiting
                continue

if __name__ == '__main__':
    main()
