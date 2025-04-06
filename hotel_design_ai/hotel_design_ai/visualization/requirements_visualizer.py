#!/usr/bin/env python3
"""
Hotel Requirements Constraints Visualizer

This script creates tables and simple visualizations of hotel space requirements
focusing on constraints like floor preferences, adjacency requirements, and separations.

Usage:
    python hotel_requirements_visualizer.py [path_to_json_file]

If no file path is provided, it will look for "hotel_requirements.json" in the current directory.
"""

import json
import sys
import os
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np
from collections import defaultdict


class HotelRequirementsVisualizer:
    """Simplified tool for visualizing hotel space constraints from JSON input file."""

    def __init__(self, json_file_path):
        """Initialize the visualizer with data from the provided JSON file."""
        self.json_file_path = json_file_path
        self.data = self._load_json_data()
        self.department_colors = {
            "public": "#4e79a7",
            "dining": "#f28e2c",
            "meeting": "#e15759",
            "recreational": "#76b7b2",
            "administrative": "#59a14f",
            "engineering": "#edc949",
            "parking": "#af7aa1",
        }
        self.output_dir = "hotel_requirements_visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_json_data(self):
        """Load and parse the JSON file."""
        try:
            with open(self.json_file_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Error: File '{self.json_file_path}' not found.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: '{self.json_file_path}' is not a valid JSON file.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            sys.exit(1)

    def visualize_all(self):
        """Generate all constraint visualizations."""
        print(f"Generating constraint visualizations for {self.json_file_path}...")

        # Create different visualizations
        self.visualize_relationships()
        self.create_floor_preference_table()
        self.create_adjacency_table()
        self.create_separation_table()
        self.create_requirements_table()

        print(f"Visualizations saved to '{self.output_dir}' directory.")

    def visualize_relationships(self):
        """Create a network graph showing adjacency and separation relationships."""
        print("Generating relationship visualization...")

        # Create directed graph
        G = nx.DiGraph()

        # Create nodes for all rooms
        for dept_name, dept_data in self.data.items():
            dept_color = self.department_colors.get(dept_name, "#cccccc")

            for space_name, space_data in dept_data.items():
                if isinstance(space_data, dict) and "room_type" in space_data:
                    room_type = space_data["room_type"]
                    node_id = f"{dept_name}:{space_name}"
                    G.add_node(
                        node_id,
                        name=space_name,
                        room_type=room_type,
                        department=dept_name,
                        color=dept_color,
                    )

        # Add edges for adjacency relationships
        for dept_name, dept_data in self.data.items():
            for space_name, space_data in dept_data.items():
                if isinstance(space_data, dict) and "requires_adjacency" in space_data:
                    source = f"{dept_name}:{space_name}"

                    for target_type in space_data["requires_adjacency"]:
                        # Find nodes of the target type
                        for node in G.nodes(data=True):
                            if node[1]["room_type"] == target_type:
                                G.add_edge(
                                    source, node[0], relationship="adjacency", weight=2
                                )

        # Add edges for separation relationships
        for dept_name, dept_data in self.data.items():
            for space_name, space_data in dept_data.items():
                if isinstance(space_data, dict) and "requires_separation" in space_data:
                    source = f"{dept_name}:{space_name}"

                    for target_type in space_data["requires_separation"]:
                        # Find nodes of the target type
                        for node in G.nodes(data=True):
                            if node[1]["room_type"] == target_type:
                                G.add_edge(
                                    source,
                                    node[0],
                                    relationship="separation",
                                    weight=1,
                                    style="dashed",
                                )

        # Create a larger figure for the network graph
        plt.figure(figsize=(20, 16))

        # Use a spring layout for node positioning
        pos = nx.spring_layout(G, k=0.15, iterations=50)

        # Draw nodes
        node_colors = [G.nodes[n]["color"] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.8)

        # Draw node labels
        labels = {n: G.nodes[n]["name"] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)

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

        plt.title("Space Relationship Diagram", fontsize=18)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "relationships.png"), dpi=200)
        plt.close()

    def create_floor_preference_table(self):
        """Create a table showing floor preferences for each space."""
        print("Creating floor preference table...")

        # Collect floor preference data
        floor_data = []

        for dept_name, dept_data in self.data.items():
            for space_name, space_data in dept_data.items():
                if isinstance(space_data, dict) and "room_type" in space_data:
                    floor = space_data.get("floor", "Not specified")
                    if floor == 0:
                        floor_label = "Ground Floor"
                    elif floor == -1:
                        floor_label = "Basement"
                    elif floor is not None:
                        floor_label = f"Floor {floor}"
                    else:
                        floor_label = "Not specified"

                    floor_data.append(
                        {
                            "Department": dept_name,
                            "Space": space_name,
                            "Room Type": space_data.get("room_type", ""),
                            "Floor": floor_label,
                            "Area (m²)": space_data.get("area", "N/A"),
                        }
                    )

        # Create DataFrame
        df = pd.DataFrame(floor_data)

        # Sort by floor
        floor_order = {
            "Basement": 0,
            "Ground Floor": 1,
            "Floor 1": 2,
            "Floor 2": 3,
            "Not specified": 9,
        }
        df["Floor Order"] = df["Floor"].map(lambda x: floor_order.get(x, 9))
        df = df.sort_values(["Floor Order", "Department", "Space"]).drop(
            "Floor Order", axis=1
        )

        # Save to CSV
        df.to_csv(os.path.join(self.output_dir, "floor_preferences.csv"), index=False)

        # Create HTML table with styling
        html = df.to_html(index=False)
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
            <div class="caption">Hotel Space Floor Preferences</div>
            {html}
        </body>
        </html>
        """

        with open(os.path.join(self.output_dir, "floor_preferences.html"), "w") as f:
            f.write(styled_html)

        # Create a floor distribution bar chart
        plt.figure(figsize=(12, 8))
        floor_counts = df["Floor"].value_counts().sort_index()

        # Reorder the index for the proper sequence (Basement, Ground Floor, Floor 1, etc.)
        ordered_index = sorted(floor_counts.index, key=lambda x: floor_order.get(x, 9))
        floor_counts = floor_counts.reindex(ordered_index)

        ax = floor_counts.plot(kind="bar", color="skyblue")
        plt.title("Number of Spaces by Floor", fontsize=16)
        plt.xlabel("Floor", fontsize=14)
        plt.ylabel("Number of Spaces", fontsize=14)
        plt.xticks(rotation=45)

        # Add value labels on top of bars
        for i, v in enumerate(floor_counts):
            ax.text(i, v + 0.5, str(v), ha="center", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "floor_distribution.png"), dpi=200)
        plt.close()

    def create_adjacency_table(self):
        """Create a table showing adjacency requirements."""
        print("Creating adjacency requirements table...")

        # Collect adjacency data
        adjacency_data = []

        for dept_name, dept_data in self.data.items():
            for space_name, space_data in dept_data.items():
                if isinstance(space_data, dict) and "requires_adjacency" in space_data:
                    adjacent_to = space_data["requires_adjacency"]

                    if adjacent_to:
                        adjacency_data.append(
                            {
                                "Department": dept_name,
                                "Space": space_name,
                                "Room Type": space_data.get("room_type", ""),
                                "Requires Adjacency To": ", ".join(adjacent_to),
                            }
                        )

        # Create DataFrame
        df = pd.DataFrame(adjacency_data)

        # Sort by department and space
        df = df.sort_values(["Department", "Space"])

        # Save to CSV
        df.to_csv(
            os.path.join(self.output_dir, "adjacency_requirements.csv"), index=False
        )

        # Create HTML table with styling
        html = df.to_html(index=False)
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
            <div class="caption">Adjacency Requirements</div>
            {html}
        </body>
        </html>
        """

        with open(
            os.path.join(self.output_dir, "adjacency_requirements.html"), "w"
        ) as f:
            f.write(styled_html)

        # Create a bar chart showing most common adjacency requirements
        if not df.empty:
            # Flatten the adjacency requirements to count frequencies
            adjacency_counts = {}
            for requirements in df["Requires Adjacency To"]:
                for req in requirements.split(", "):
                    if req in adjacency_counts:
                        adjacency_counts[req] += 1
                    else:
                        adjacency_counts[req] = 1

            # Create a bar chart
            plt.figure(figsize=(12, 8))
            adj_types = list(adjacency_counts.keys())
            counts = list(adjacency_counts.values())

            # Sort by frequency
            sorted_indices = np.argsort(counts)[::-1]
            adj_types = [adj_types[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]

            plt.bar(adj_types, counts, color="green", alpha=0.7)
            plt.title("Most Common Adjacency Requirements", fontsize=16)
            plt.xlabel("Room Type", fontsize=14)
            plt.ylabel("Number of Spaces Requiring Adjacency", fontsize=14)
            plt.xticks(rotation=45, ha="right")

            # Add value labels on top of bars
            for i, v in enumerate(counts):
                plt.text(i, v + 0.1, str(v), ha="center", fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "adjacency_counts.png"), dpi=200)
            plt.close()

    def create_separation_table(self):
        """Create a table showing separation requirements."""
        print("Creating separation requirements table...")

        # Collect separation data
        separation_data = []

        for dept_name, dept_data in self.data.items():
            for space_name, space_data in dept_data.items():
                if isinstance(space_data, dict) and "requires_separation" in space_data:
                    separate_from = space_data["requires_separation"]

                    if separate_from:
                        separation_data.append(
                            {
                                "Department": dept_name,
                                "Space": space_name,
                                "Room Type": space_data.get("room_type", ""),
                                "Requires Separation From": ", ".join(separate_from),
                            }
                        )

        # Create DataFrame
        df = pd.DataFrame(separation_data)

        # Sort by department and space
        df = df.sort_values(["Department", "Space"])

        # Save to CSV
        df.to_csv(
            os.path.join(self.output_dir, "separation_requirements.csv"), index=False
        )

        # Create HTML table with styling
        html = df.to_html(index=False)
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
            <div class="caption">Separation Requirements</div>
            {html}
        </body>
        </html>
        """

        with open(
            os.path.join(self.output_dir, "separation_requirements.html"), "w"
        ) as f:
            f.write(styled_html)

        # Create a bar chart showing most common separation requirements
        if not df.empty:
            # Flatten the separation requirements to count frequencies
            separation_counts = {}
            for requirements in df["Requires Separation From"]:
                for req in requirements.split(", "):
                    if req in separation_counts:
                        separation_counts[req] += 1
                    else:
                        separation_counts[req] = 1

            # Create a bar chart
            plt.figure(figsize=(12, 8))
            sep_types = list(separation_counts.keys())
            counts = list(separation_counts.values())

            # Sort by frequency
            sorted_indices = np.argsort(counts)[::-1]
            sep_types = [sep_types[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]

            plt.bar(sep_types, counts, color="red", alpha=0.7)
            plt.title("Most Common Separation Requirements", fontsize=16)
            plt.xlabel("Room Type", fontsize=14)
            plt.ylabel("Number of Spaces Requiring Separation", fontsize=14)
            plt.xticks(rotation=45, ha="right")

            # Add value labels on top of bars
            for i, v in enumerate(counts):
                plt.text(i, v + 0.1, str(v), ha="center", fontsize=12)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "separation_counts.png"), dpi=200)
            plt.close()

    def create_requirements_table(self):
        """Create a table showing special requirements."""
        print("Creating special requirements table...")

        # Collect requirements data
        requirements_data = []

        for dept_name, dept_data in self.data.items():
            for space_name, space_data in dept_data.items():
                if isinstance(space_data, dict) and "room_type" in space_data:
                    requirements = []

                    if space_data.get("requires_natural_light", False):
                        requirements.append("Natural Light")

                    if space_data.get("requires_column_free", False):
                        requirements.append("Column-Free")

                    if space_data.get("requires_exhaust", False):
                        requirements.append("Exhaust")

                    if space_data.get("requires_column_spacing", False):
                        requirements.append("Column Spacing")

                    requirements_data.append(
                        {
                            "Department": dept_name,
                            "Space": space_name,
                            "Room Type": space_data.get("room_type", ""),
                            "Area (m²)": space_data.get("area", "N/A"),
                            "Natural Light": (
                                "Yes"
                                if space_data.get("requires_natural_light", False)
                                else "No"
                            ),
                            "Column-Free": (
                                "Yes"
                                if space_data.get("requires_column_free", False)
                                else "No"
                            ),
                            "Exhaust": (
                                "Yes"
                                if space_data.get("requires_exhaust", False)
                                else "No"
                            ),
                            "Column Spacing": (
                                "Yes"
                                if space_data.get("requires_column_spacing", False)
                                else "No"
                            ),
                            "Requirements": (
                                ", ".join(requirements) if requirements else "None"
                            ),
                        }
                    )

        # Create DataFrame
        df = pd.DataFrame(requirements_data)

        # Sort by department and space
        df = df.sort_values(["Department", "Space"])

        # Save to CSV
        df.to_csv(
            os.path.join(self.output_dir, "special_requirements.csv"), index=False
        )

        # Create HTML table with styling
        html = df.to_html(index=False)
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
            <div class="caption">Special Requirements</div>
            {html}
        </body>
        </html>
        """

        with open(os.path.join(self.output_dir, "special_requirements.html"), "w") as f:
            f.write(styled_html)

        # Create a bar chart showing requirement counts
        plt.figure(figsize=(10, 6))

        req_counts = {
            "Natural Light": df["Natural Light"].value_counts().get("Yes", 0),
            "Column-Free": df["Column-Free"].value_counts().get("Yes", 0),
            "Exhaust": df["Exhaust"].value_counts().get("Yes", 0),
            "Column Spacing": df["Column Spacing"].value_counts().get("Yes", 0),
        }

        req_types = list(req_counts.keys())
        counts = list(req_counts.values())

        plt.bar(req_types, counts, color="blue", alpha=0.7)
        plt.title("Special Requirements Count", fontsize=16)
        plt.xlabel("Requirement Type", fontsize=14)
        plt.ylabel("Number of Spaces", fontsize=14)

        # Add value labels on top of bars
        for i, v in enumerate(counts):
            plt.text(i, v + 0.5, str(v), ha="center", fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "requirements_counts.png"), dpi=200)
        plt.close()


def main():
    # Get the JSON file path from command line argument or use default
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = "/Users/ky/01_Projects/hotel_desgin_ai/hotel_design_ai/data/program/hotel_requirements.json"
        if not os.path.exists(json_file_path):
            print(
                f"Error: Default file '{json_file_path}' not found. Please provide a path to a valid JSON file."
            )
            print("Usage: python hotel_requirements_visualizer.py [path_to_json_file]")
            sys.exit(1)

    # Initialize and run the visualizer
    visualizer = HotelRequirementsVisualizer(json_file_path)
    visualizer.visualize_all()

    print("All visualizations completed successfully.")
    print(f"Open the '{visualizer.output_dir}' directory to view the outputs.")


if __name__ == "__main__":
    main()
