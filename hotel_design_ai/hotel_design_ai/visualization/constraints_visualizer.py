#!/usr/bin/env python3
"""
Hotel Constraints Visualizer

This script visualizes hotel design constraints from the data/constraints directory,
helping users understand the architectural rules applied by the Hotel Design AI system.

Usage:
    python hotel_constraints_visualizer.py [constraints_dir_path]

If no directory path is provided, it will look for "data/constraints" in the current directory.
"""

import json
import sys
import os
import glob
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict


class HotelConstraintsVisualizer:
    """Tool for visualizing hotel design constraints from JSON files."""

    def __init__(self, constraints_dir):
        """Initialize the visualizer with the constraints directory."""
        self.constraints_dir = constraints_dir
        self.adjacency_constraints = []
        self.separation_constraints = []
        self.floor_constraints = []
        self.exterior_constraints = []
        self.design_constraints = {}

        # Color scheme for visualizations
        self.constraint_colors = {
            "adjacency": "#4e79a7",
            "separation": "#e15759",
            "floor": "#59a14f",
            "exterior": "#f28e2c",
            "design": "#76b7b2",
        }

        # Load all constraint data
        self._load_constraints()

        # Output directory
        self.output_dir = "hotel_constraints_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_constraints(self):
        """Load constraints from all JSON files in the constraints directory."""
        # Check if constraints directory exists
        if not os.path.exists(self.constraints_dir):
            print(f"Error: Constraints directory '{self.constraints_dir}' not found.")
            sys.exit(1)

        # Load adjacency constraints
        adjacency_path = os.path.join(self.constraints_dir, "adjacency.json")
        if os.path.exists(adjacency_path):
            try:
                with open(adjacency_path, "r") as f:
                    self.adjacency_constraints = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse '{adjacency_path}'. File may be corrupted."
                )

        # Load separation constraints
        separation_path = os.path.join(self.constraints_dir, "separation.json")
        if os.path.exists(separation_path):
            try:
                with open(separation_path, "r") as f:
                    self.separation_constraints = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse '{separation_path}'. File may be corrupted."
                )

        # Load floor constraints
        floor_path = os.path.join(self.constraints_dir, "floor.json")
        if os.path.exists(floor_path):
            try:
                with open(floor_path, "r") as f:
                    self.floor_constraints = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse '{floor_path}'. File may be corrupted."
                )

        # Load exterior constraints
        exterior_path = os.path.join(self.constraints_dir, "exterior.json")
        if os.path.exists(exterior_path):
            try:
                with open(exterior_path, "r") as f:
                    self.exterior_constraints = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse '{exterior_path}'. File may be corrupted."
                )

        # Load design constraints
        design_path = os.path.join(self.constraints_dir, "design.json")
        if os.path.exists(design_path):
            try:
                with open(design_path, "r") as f:
                    self.design_constraints = json.load(f)
            except json.JSONDecodeError:
                print(
                    f"Warning: Could not parse '{design_path}'. File may be corrupted."
                )

    def visualize_all(self):
        """Generate all constraint visualizations."""
        print(f"Generating constraint visualizations for '{self.constraints_dir}'...")

        # Visualize relationship constraints
        self.visualize_relationship_constraints()

        # Visualize floor constraints
        self.visualize_floor_constraints()

        # Visualize exterior constraints
        self.visualize_exterior_constraints()

        # Visualize design constraints
        self.visualize_design_constraints()

        # Create constraint summary
        self.create_constraint_summary()

        print(f"Visualizations saved to '{self.output_dir}' directory.")

    def visualize_relationship_constraints(self):
        """Create visualizations for adjacency and separation constraints."""
        print("Generating relationship constraints visualization...")

        # Create directed graph
        G = nx.DiGraph()

        # Get all unique room types from all constraints
        room_types = set()
        for constraint in self.adjacency_constraints + self.separation_constraints:
            if "room_type1" in constraint and "room_type2" in constraint:
                room_types.add(constraint["room_type1"])
                room_types.add(constraint["room_type2"])

        # Create nodes for all room types
        for room_type in room_types:
            G.add_node(room_type, name=room_type)

        # Add edges for adjacency constraints
        for constraint in self.adjacency_constraints:
            if "room_type1" in constraint and "room_type2" in constraint:
                G.add_edge(
                    constraint["room_type1"],
                    constraint["room_type2"],
                    relationship="adjacency",
                    weight=constraint.get("weight", 1.0),
                    description=constraint.get("description", ""),
                )

        # Add edges for separation constraints
        for constraint in self.separation_constraints:
            if "room_type1" in constraint and "room_type2" in constraint:
                G.add_edge(
                    constraint["room_type1"],
                    constraint["room_type2"],
                    relationship="separation",
                    weight=constraint.get("weight", 1.0),
                    description=constraint.get("description", ""),
                )

        # Create a larger figure for the network graph
        plt.figure(figsize=(20, 16))

        # Use a spring layout for node positioning
        pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, alpha=0.8)

        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=10)

        # Draw adjacency edges
        adjacency_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d.get("relationship") == "adjacency"
        ]
        nx.draw_networkx_edges(
            G, pos, edgelist=adjacency_edges, edge_color="green", arrows=True
        )

        # Draw separation edges
        separation_edges = [
            (u, v)
            for u, v, d in G.edges(data=True)
            if d.get("relationship") == "separation"
        ]
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=separation_edges,
            edge_color="red",
            style="dashed",
            arrows=True,
        )

        # Add legend
        plt.plot([0], [0], "-", color="green", label="Requires Adjacency")
        plt.plot([0], [0], "--", color="red", label="Requires Separation")
        plt.legend(fontsize=12)

        plt.title("Relationship Constraints Between Room Types", fontsize=18)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "relationship_constraints.png"), dpi=200
        )
        plt.close()

        # Create an adjacency matrix for all constraints
        constraint_tables = []

        # Create tables for adjacency constraints
        if self.adjacency_constraints:
            # Convert to DataFrame
            adj_df = pd.DataFrame(self.adjacency_constraints)

            # Sort by weight (descending) and room types
            adj_df = adj_df.sort_values(
                by=["weight", "room_type1", "room_type2"], ascending=[False, True, True]
            )

            # Create HTML table
            html = adj_df.to_html(index=False)
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #ddd; }}
                    .caption {{ font-size: 1.5em; font-weight: bold; margin-bottom: 10px; }}
                </style>
            </head>
            <body>
                <div class="caption">Adjacency Constraints</div>
                {html}
            </body>
            </html>
            """

            with open(
                os.path.join(self.output_dir, "adjacency_constraints.html"), "w"
            ) as f:
                f.write(styled_html)

            # Save to CSV
            adj_df.to_csv(
                os.path.join(self.output_dir, "adjacency_constraints.csv"), index=False
            )

            constraint_tables.append(("Adjacency Constraints", adj_df))

        # Create tables for separation constraints
        if self.separation_constraints:
            # Convert to DataFrame
            sep_df = pd.DataFrame(self.separation_constraints)

            # Sort by weight (descending) and room types
            sep_df = sep_df.sort_values(
                by=["weight", "room_type1", "room_type2"], ascending=[False, True, True]
            )

            # Create HTML table
            html = sep_df.to_html(index=False)
            styled_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                    th {{ background-color: #E74C3C; color: white; }}
                    tr:nth-child(even) {{ background-color: #f2f2f2; }}
                    tr:hover {{ background-color: #ddd; }}
                    .caption {{ font-size: 1.5em; font-weight: bold; margin-bottom: 10px; }}
                </style>
            </head>
            <body>
                <div class="caption">Separation Constraints</div>
                {html}
            </body>
            </html>
            """

            with open(
                os.path.join(self.output_dir, "separation_constraints.html"), "w"
            ) as f:
                f.write(styled_html)

            # Save to CSV
            sep_df.to_csv(
                os.path.join(self.output_dir, "separation_constraints.csv"), index=False
            )

            constraint_tables.append(("Separation Constraints", sep_df))

        # Create a weighted adjacency heatmap
        if room_types:
            # Sort room types alphabetically for consistency
            sorted_room_types = sorted(room_types)
            n = len(sorted_room_types)

            # Create an adjacency matrix
            matrix = np.zeros((n, n))

            # Map from room type to index
            room_to_idx = {room: i for i, room in enumerate(sorted_room_types)}

            # Fill matrix with weights
            for constraint in self.adjacency_constraints:
                if "room_type1" in constraint and "room_type2" in constraint:
                    r1 = constraint["room_type1"]
                    r2 = constraint["room_type2"]
                    if r1 in room_to_idx and r2 in room_to_idx:
                        weight = constraint.get("weight", 1.0)
                        i, j = room_to_idx[r1], room_to_idx[r2]
                        matrix[i, j] = weight
                        matrix[j, i] = weight  # Make symmetric

            # Create a second matrix for separation constraints (negative weights)
            for constraint in self.separation_constraints:
                if "room_type1" in constraint and "room_type2" in constraint:
                    r1 = constraint["room_type1"]
                    r2 = constraint["room_type2"]
                    if r1 in room_to_idx and r2 in room_to_idx:
                        weight = -constraint.get(
                            "weight", 1.0
                        )  # Negative for separation
                        i, j = room_to_idx[r1], room_to_idx[r2]
                        matrix[i, j] = weight
                        matrix[j, i] = weight  # Make symmetric

            # Create heatmap
            plt.figure(figsize=(14, 12))
            plt.imshow(matrix, cmap="RdBu_r", vmin=-3, vmax=3)

            # Add labels
            plt.xticks(range(n), sorted_room_types, rotation=90)
            plt.yticks(range(n), sorted_room_types)

            # Add colorbar
            cbar = plt.colorbar()
            cbar.set_label(
                "Relationship Strength (negative = separation, positive = adjacency)"
            )

            # Add grid
            plt.grid(False)

            plt.title("Room Relationships Matrix", fontsize=16)
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.output_dir, "relationship_matrix.png"), dpi=200
            )
            plt.close()

    def visualize_floor_constraints(self):
        """Create visualizations for floor constraints."""
        print("Generating floor constraints visualization...")

        if not self.floor_constraints:
            print("No floor constraints found. Skipping floor visualization.")
            return

        # Create DataFrame
        floor_df = pd.DataFrame(self.floor_constraints)

        # Sort by floor and weight
        floor_df = floor_df.sort_values(by=["floor", "weight"], ascending=[True, False])

        # Save to CSV and HTML
        floor_df.to_csv(
            os.path.join(self.output_dir, "floor_constraints.csv"), index=False
        )

        # Create HTML table
        html = floor_df.to_html(index=False)
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #3498DB; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #ddd; }}
                .caption {{ font-size: 1.5em; font-weight: bold; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="caption">Floor Constraints</div>
            {html}
        </body>
        </html>
        """

        with open(os.path.join(self.output_dir, "floor_constraints.html"), "w") as f:
            f.write(styled_html)

        # Create a visualization of floor assignments
        floor_counts = floor_df["floor"].value_counts().sort_index()

        # Map floor numbers to labels
        floor_labels = {}
        for floor in floor_counts.index:
            if floor == 0:
                floor_labels[floor] = "Ground Floor"
            elif floor < 0:
                floor_labels[floor] = f"Basement {abs(floor)}"
            else:
                floor_labels[floor] = f"Floor {floor}"

        plt.figure(figsize=(10, 8))
        bars = plt.bar(
            [floor_labels.get(f, f) for f in floor_counts.index],
            floor_counts.values,
            color=self.constraint_colors["floor"],
        )

        plt.title("Room Type Assignments by Floor", fontsize=16)
        plt.xlabel("Floor", fontsize=14)
        plt.ylabel("Number of Room Types", fontsize=14)
        plt.xticks(rotation=45, ha="right")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1,
                str(int(height)),
                ha="center",
                fontsize=12,
            )

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "floor_assignments.png"), dpi=200)
        plt.close()

        # Create a visualization of weights by room type and floor
        plt.figure(figsize=(14, 10))

        # Group by floor
        for floor in sorted(floor_df["floor"].unique()):
            floor_data = floor_df[floor_df["floor"] == floor]
            weights = floor_data["weight"]
            room_types = floor_data["room_type"]

            floor_label = floor_labels.get(floor, f"Floor {floor}")
            plt.scatter(room_types, weights, label=floor_label, s=100, alpha=0.7)

        plt.title("Floor Constraint Weights by Room Type", fontsize=16)
        plt.xlabel("Room Type", fontsize=14)
        plt.ylabel("Weight", fontsize=14)
        plt.xticks(rotation=90)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Floor")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "floor_constraint_weights.png"), dpi=200
        )
        plt.close()

    def visualize_exterior_constraints(self):
        """Create visualizations for exterior access constraints."""
        print("Generating exterior constraints visualization...")

        if not self.exterior_constraints:
            print("No exterior constraints found. Skipping exterior visualization.")
            return

        # Create DataFrame
        exterior_df = pd.DataFrame(self.exterior_constraints)

        # Sort by weight (descending)
        exterior_df = exterior_df.sort_values(by="weight", ascending=False)

        # Save to CSV and HTML
        exterior_df.to_csv(
            os.path.join(self.output_dir, "exterior_constraints.csv"), index=False
        )

        # Create HTML table
        html = exterior_df.to_html(index=False)
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #F39C12; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #ddd; }}
                .caption {{ font-size: 1.5em; font-weight: bold; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="caption">Exterior Access Constraints</div>
            {html}
        </body>
        </html>
        """

        with open(os.path.join(self.output_dir, "exterior_constraints.html"), "w") as f:
            f.write(styled_html)

        # Create a bar chart of exterior constraints by weight
        plt.figure(figsize=(12, 8))

        # Sort by weight for visualization
        exterior_df = exterior_df.sort_values(by="weight")

        bars = plt.barh(
            exterior_df["room_type"],
            exterior_df["weight"],
            color=self.constraint_colors["exterior"],
            alpha=0.7,
        )

        plt.title("Exterior Access Requirements by Room Type", fontsize=16)
        plt.xlabel("Constraint Weight", fontsize=14)
        plt.ylabel("Room Type", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7, axis="x")

        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(
                width + 0.05,
                bar.get_y() + bar.get_height() / 2.0,
                f"{width:.1f}",
                va="center",
                fontsize=10,
            )

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "exterior_constraints.png"), dpi=200)
        plt.close()

    def visualize_design_constraints(self):
        """Create visualizations for design constraints."""
        print("Generating design constraints visualization...")

        if not self.design_constraints:
            print("No design constraints found. Skipping design visualization.")
            return

        # Create HTML summary of design constraints
        html_parts = ["<h1>Design Constraints Summary</h1>"]

        for category, constraints in self.design_constraints.items():
            html_parts.append(f"<h2>{category.title()}</h2>")

            if isinstance(constraints, dict):
                # Convert dictionary to HTML table
                html_parts.append(
                    '<table border="1" style="border-collapse: collapse; width: 100%;">'
                )

                # Add table header
                html_parts.append(
                    '<tr style="background-color: #76b7b2; color: white;">'
                )
                html_parts.append(
                    '<th style="padding: 8px; text-align: left;">Property</th>'
                )
                html_parts.append(
                    '<th style="padding: 8px; text-align: left;">Value</th>'
                )
                html_parts.append("</tr>")

                # Add table rows
                for prop, value in constraints.items():
                    html_parts.append('<tr style="background-color: #f2f2f2;">')
                    html_parts.append(f'<td style="padding: 8px;">{prop}</td>')

                    # Format value based on type
                    if isinstance(value, list):
                        value_str = ", ".join(str(v) for v in value)
                    elif isinstance(value, dict):
                        value_str = "<ul>"
                        for k, v in value.items():
                            if isinstance(v, list):
                                v_str = ", ".join(str(item) for item in v)
                            else:
                                v_str = str(v)
                            value_str += f"<li><strong>{k}</strong>: {v_str}</li>"
                        value_str += "</ul>"
                    else:
                        value_str = str(value)

                    html_parts.append(f'<td style="padding: 8px;">{value_str}</td>')
                    html_parts.append("</tr>")

                html_parts.append("</table>")
                html_parts.append("<br>")
            else:
                # For simple values
                html_parts.append(f"<p>{constraints}</p>")

        # Combine all parts into a complete HTML document
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333366; }}
                h2 {{ color: #336699; margin-top: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #76b7b2; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #ddd; }}
            </style>
        </head>
        <body>
            {"".join(html_parts)}
        </body>
        </html>
        """

        with open(os.path.join(self.output_dir, "design_constraints.html"), "w") as f:
            f.write(styled_html)

        # Extract key values to visualize
        # For example, visualize structural requirements
        if "structural" in self.design_constraints:
            structural = self.design_constraints["structural"]

            if "column_free_spaces" in structural and isinstance(
                structural["column_free_spaces"], list
            ):
                # Create a visualization of column-free spaces
                plt.figure(figsize=(10, 6))
                spaces = structural["column_free_spaces"]

                y_pos = range(len(spaces))
                plt.barh(
                    y_pos,
                    [1] * len(spaces),
                    color=self.constraint_colors["design"],
                    alpha=0.7,
                )
                plt.yticks(y_pos, spaces)
                plt.xlabel("Requirement")
                plt.title("Spaces Requiring Column-Free Design")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(self.output_dir, "column_free_spaces.png"), dpi=200
                )
                plt.close()

    def create_constraint_summary(self):
        """Create a summary of all constraints."""
        print("Creating constraint summary...")

        # Count constraints by type
        constraint_counts = {
            "Adjacency": len(self.adjacency_constraints),
            "Separation": len(self.separation_constraints),
            "Floor": len(self.floor_constraints),
            "Exterior": len(self.exterior_constraints),
        }

        # Create a bar chart of constraint counts
        plt.figure(figsize=(10, 6))

        bars = plt.bar(
            constraint_counts.keys(),
            constraint_counts.values(),
            color=[
                self.constraint_colors["adjacency"],
                self.constraint_colors["separation"],
                self.constraint_colors["floor"],
                self.constraint_colors["exterior"],
            ],
            alpha=0.7,
        )

        plt.title("Hotel Design Constraints by Type", fontsize=16)
        plt.ylabel("Number of Constraints", fontsize=14)
        plt.grid(True, linestyle="--", alpha=0.7, axis="y")

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.5,
                str(int(height)),
                ha="center",
                fontsize=12,
            )

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "constraint_summary.png"), dpi=200)
        plt.close()

        # Create a summary table with all constraint counts and top constraints
        summary_data = []

        # Add adjacency constraints summary
        if self.adjacency_constraints:
            top_adjacency = sorted(
                self.adjacency_constraints,
                key=lambda x: x.get("weight", 0),
                reverse=True,
            )[:3]
            top_str = ", ".join(
                [
                    f"{c['room_type1']}-{c['room_type2']} ({c.get('weight', 1.0)})"
                    for c in top_adjacency
                ]
            )
            summary_data.append(
                {
                    "Constraint Type": "Adjacency",
                    "Count": len(self.adjacency_constraints),
                    "Top Constraints (by weight)": top_str,
                }
            )

        # Add separation constraints summary
        if self.separation_constraints:
            top_separation = sorted(
                self.separation_constraints,
                key=lambda x: x.get("weight", 0),
                reverse=True,
            )[:3]
            top_str = ", ".join(
                [
                    f"{c['room_type1']}-{c['room_type2']} ({c.get('weight', 1.0)})"
                    for c in top_separation
                ]
            )
            summary_data.append(
                {
                    "Constraint Type": "Separation",
                    "Count": len(self.separation_constraints),
                    "Top Constraints (by weight)": top_str,
                }
            )

        # Add floor constraints summary
        if self.floor_constraints:
            top_floor = sorted(
                self.floor_constraints, key=lambda x: x.get("weight", 0), reverse=True
            )[:3]
            top_str = ", ".join(
                [
                    f"{c['room_type']} (Floor {c['floor']}, {c.get('weight', 1.0)})"
                    for c in top_floor
                ]
            )
            summary_data.append(
                {
                    "Constraint Type": "Floor",
                    "Count": len(self.floor_constraints),
                    "Top Constraints (by weight)": top_str,
                }
            )

        # Add exterior constraints summary
        if self.exterior_constraints:
            top_exterior = sorted(
                self.exterior_constraints,
                key=lambda x: x.get("weight", 0),
                reverse=True,
            )[:3]
            top_str = ", ".join(
                [f"{c['room_type']} ({c.get('weight', 1.0)})" for c in top_exterior]
            )
            summary_data.append(
                {
                    "Constraint Type": "Exterior",
                    "Count": len(self.exterior_constraints),
                    "Top Constraints (by weight)": top_str,
                }
            )

        # Convert to DataFrame
        summary_df = pd.DataFrame(summary_data)

        # Save to CSV
        summary_df.to_csv(
            os.path.join(self.output_dir, "constraint_summary.csv"), index=False
        )

        # Create HTML table
        html = summary_df.to_html(index=False)
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
                th {{ background-color: #333366; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                tr:hover {{ background-color: #ddd; }}
                .caption {{ font-size: 1.5em; font-weight: bold; margin-bottom: 10px; }}
            </style>
        </head>
        <body>
            <div class="caption">Constraint Summary</div>
            {html}
        </body>
        </html>
        """

        with open(os.path.join(self.output_dir, "constraint_summary.html"), "w") as f:
            f.write(styled_html)


def main():
    # Get the constraints directory path from command line argument or use default
    if len(sys.argv) > 1:
        constraints_dir = sys.argv[1]
    else:
        # Try to find data/constraints directory
        if os.path.exists("data/constraints"):
            constraints_dir = "data/constraints"
        else:
            print(
                "Error: Could not find default 'data/constraints' directory. Please provide a path to the constraints directory."
            )
            print(
                "Usage: python hotel_constraints_visualizer.py [constraints_dir_path]"
            )
            sys.exit(1)

    # Initialize and run the visualizer
    visualizer = HotelConstraintsVisualizer(constraints_dir)
    visualizer.visualize_all()

    print("All constraint visualizations completed successfully.")
    print(f"Open the '{visualizer.output_dir}' directory to view the outputs.")


if __name__ == "__main__":
    main()
