# Hotel Design AI: Development and Optimization Pipeline

This document outlines the recommended pipeline for running, improving, and optimizing the Hotel Design AI system.

## Getting Started Pipeline

### 1. Setup Project Environment

```bash
# Clone repository (if using version control)
git clone <repository-url>
cd hotel-design-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Create necessary output directory
mkdir -p output
```

### 2. Initial Project Configuration

1. **Update Building Envelope**:
   - Open `hotel_design_ai/config/ENV.py`
   - Modify `BUILDING_ENVELOPE` variables to match your project dimensions
   - Adjust structural grid spacing as needed (typically 7.5-9m for hotels)

2. **Define Room Program**:
   - Update `PROGRAM_REQUIREMENTS` in `ENV.py` with your specific room requirements
   - Or edit JSON files in `data/room_types/` directory

3. **Set Constraints**:
   - Review and modify `data/constraints/standard_hotel.json`
   - Uncomment `update_adjacencies_from_constraints()` in `ENV.py` to load from this file

### 3. Generate Initial Layout

```bash
# Generate using rule-based algorithm
python main.py --mode rule --visualize

# Save outputs to specific directory
python main.py --mode rule --output ./my_project
```

### 4. Review and Iterate

1. **Analyze Initial Layout**:
   - Review the visualization outputs in the output directory
   - Check layout metrics for space utilization, adjacency satisfaction, etc.

2. **Fix Key Rooms**:
   - Identify key rooms to fix in place (e.g., entrance, lobby, vertical circulation)
   - Create a JSON file with fixed positions (example below)

   ```json
   {
     "1": [30.0, 0.0, 0.0],
     "2": [30.0, 10.0, 0.0],
     "5": [20.0, 20.0, 0.0]
   }
   ```

3. **Regenerate with Fixed Rooms**:
   ```bash
   python main.py --mode rule --fixed-rooms my_fixed_rooms.json
   ```

## Optimization Pipeline

### 1. Train RL Model

```bash
# Train RL model with simulated feedback
python main.py --mode rl --train-iterations 20 --simulate-feedback --rl-model models/hotel_rl.pt

# Train longer for better results
python main.py --mode rl --train-iterations 50 --simulate-feedback --rl-model models/hotel_rl.pt
```

### 2. Use Hybrid Approach

```bash
# Generate using hybrid approach (rule-based + RL refinement)
python main.py --mode hybrid --rl-model models/hotel_rl.pt --visualize
```

### 3. Constraint Fine-Tuning

1. **Adjust Constraint Weights**:
   - Edit weights in `data/constraints/standard_hotel.json`
   - Higher weights (e.g., 2.0-3.0) for critical constraints
   - Lower weights (e.g., 0.5-1.0) for preferences

2. **Add Custom Constraints**:
   - Create department-specific constraint files (e.g., `dining_constraints.json`)
   - Add specific adjacency and separation rules

3. **Regenerate with Custom Constraints**:
   ```bash
   python main.py --mode hybrid --constraints data/constraints/my_custom_constraints.json
   ```

### 4. Multiple Iteration Comparison

```bash
# Generate multiple layouts with different seeds
python main.py --mode hybrid --seed 42 --output ./iteration1
python main.py --mode hybrid --seed 123 --output ./iteration2
python main.py --mode hybrid --seed 7 --output ./iteration3

# Compare layouts visually and using metrics
```

## Advanced Optimization Techniques

### 1. Progressive Refinement

1. **Start with Public Areas**:
   - First focus on entrance, lobby, and major public spaces
   - Fix these in place once satisfied

2. **Add Major Program Elements**:
   - Add restaurant, meeting spaces, and amenities
   - Optimize their positions in relation to public areas

3. **Add Back-of-House**:
   - Add kitchen, service areas, admin offices
   - Ensure proper service connections and separation

4. **Add Guest Rooms Last**:
   - Add guest rooms and optimize floor layouts
   - Test different room arrangements (single vs. double-loaded corridor)

### 2. Template-Based Approach

1. **Create Floor Templates**:
   - Build templates for typical floors in `data/templates/`
   - Create variants (L-shaped, double-loaded, atrium, etc.)

2. **Apply Templates**:
   - Modify `main.py` to load and apply templates
   - Stack different templates for different floors

3. **Custom Vertical Stacking**:
   - Ensure proper alignment of structural elements
   - Check for consistent vertical circulation

### 3. Performance Optimization

#### Code Optimization

1. **Profile Code Performance**:
   ```bash
   python -m cProfile -o profile_results.prof main.py --mode rule
   ```

2. **Analyze Profile Results**:
   ```bash
   python -m pstats profile_results.prof
   ```

3. **Optimize Bottlenecks**:
   - Common areas for optimization:
     - Spatial grid operations (use NumPy vectorization)
     - Collision detection algorithms
     - Path finding functions
     - Adjacency checking

#### Algorithm Optimization

1. **Adjust Grid Resolution**:
   - Increase `grid_size` in `ENV.py` for faster but less precise layouts
   - Typical range: 0.5m (detailed) to 2.0m (schematic)

2. **Use Hierarchical Approach**:
   - First optimize at coarse grid (zoning)
   - Then refine at finer grid (detailed placement)

3. **Enable GPU Acceleration** (for RL):
   - Update RL engine to use GPU if available:
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   ```

## Export and Integration Pipeline

### 1. Export to Design Tools

```bash
# Export to Revit-compatible format
python main.py --mode hybrid --export-formats revit

# Export for web visualization
python main.py --mode hybrid --export-formats threejs
```

### 2. Post-Processing Optimization

1. **Align to Building Grid**:
   - Adjust room positions to align perfectly with structural grid
   - Fix dimensions to standard modular sizes

2. **Refine Circulation**:
   - Analyze and optimize corridor widths and paths
   - Ensure proper egress requirements

3. **Manual Design Refinement**:
   - Import to CAD/BIM software for detailed design
   - Adjust for specific building systems integration

### 3. Parameterization for Mass Customization

1. **Create Parameter Sets**:
   - Define parameter ranges for key variables (e.g., building width, depth)
   - Create configuration files for different building types

2. **Batch Generation**:
   - Write scripts to generate multiple variants
   - Compare results across parameter ranges

3. **Sensitivity Analysis**:
   - Identify which parameters most affect layout quality
   - Focus optimization on these key parameters

## Continuous Improvement Pipeline

### 1. System Improvement

1. **Enhance RL Algorithm**:
   - Implement more sophisticated RL techniques (A3C, PPO)
   - Add experience replay buffer for better learning

2. **Extend Constraint System**:
   - Add new constraint types (e.g., view analysis, solar access)
   - Implement soft vs. hard constraints

3. **Improve Visualization**:
   - Add interactive 3D viewer using three.js
   - Create dashboard for comparing layouts

### 2. Data Collection and Learning

1. **Collect User Feedback**:
   - Track which layouts users prefer
   - Record specific feedback on room placements

2. **Build Training Dataset**:
   - Accumulate successful layouts
   - Use to pre-train RL models

3. **Implement Transfer Learning**:
   - Train on general hotel layouts
   - Fine-tune for specific hotel types

### 3. Expanding Capabilities

1. **Multi-Building Planning**:
   - Extend to site planning with multiple buildings
   - Include landscape and circulation

2. **Vertical Transportation Optimization**:
   - Add elevator traffic analysis
   - Optimize core locations and counts

3. **Energy and Sustainability**:
   - Add energy performance metrics
   - Optimize orientation and envelope design

4. **Cost Optimization**:
   - Add construction cost estimation
   - Optimize for cost-efficiency

## Development Best Practices

1. **Version Control**:
   - Commit code changes with meaningful messages
   - Create branches for major features

2. **Testing**:
   - Add unit tests for core components
   - Add integration tests for end-to-end workflow

3. **Documentation**:
   - Keep README updated
   - Document key functions and algorithms
   - Create user guide for non-technical stakeholders

4. **Refactoring**:
   - Regular code clean-up
   - Improve modularity and reusability
