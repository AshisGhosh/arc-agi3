# ARC-AGI-3 Game Mechanics Analysis

## Executive Summary

ARC-AGI-3 is fundamentally different from static grid transformations (ARC-AGI-1/2). It requires:
- **Interactive exploration** to discover hidden rules
- **Multi-step planning** with limited resources
- **State tracking** across actions
- **Diverse mechanics** per game (navigation, clicking, pattern matching)

## Available Games (Preview)

| Game | Type | Actions | Levels | Core Mechanic |
|------|------|---------|--------|---------------|
| ls20 | Navigation | 1-4 (UDLR) | 7 | Key-lock matching with shape/color/rotation |
| vc33 | Click | 6 (x,y) | 7 | Budget-based cell manipulation |
| ft09 | Click | 1-6 | 6 | Color cycling to satisfy constraints |

## Detailed Mechanics

### ls20 (Locksmith)
**Type:** Spatial navigation + pattern matching

**Actions:**
- 1 = Move up (5 pixels)
- 2 = Move down (5 pixels)
- 3 = Move left (5 pixels)
- 4 = Move right (5 pixels)

**Mechanics:**
- Navigate a sprite through a grid world
- Interact with objects: keys, locks, switches
- Configure key properties: shape, color, rotation
- Match key configuration to each lock
- 3 lives - wrong moves reset to checkpoint

**Win condition:** Unlock all locks with correct key configurations

**Skills required:**
- Spatial navigation (pathfinding)
- Pattern matching (key→lock)
- State tracking (current vs required config)
- Multi-step planning

---

### vc33 (Budget Logic)
**Type:** Click-based constraint puzzle

**Actions:**
- 6 = Click at (x, y) coordinate

**Mechanics:**
- Click cells to toggle/modify state
- Limited budget for actions
- Must satisfy logical constraints

**Win condition:** All constraints satisfied within budget

**Skills required:**
- Constraint satisfaction
- State propagation
- Resource optimization

---

### ft09 (Pattern Matching)
**Type:** Click-based color cycling

**Actions:**
- 1-5 = Predefined pattern effects
- 6 = Click at (x, y) coordinate

**Mechanics:**
- Grid cells have colors
- Clicking cycles colors through a sequence
- Pattern templates affect multiple cells
- Target: match color constraints (same/different rules)

**Win condition:** All color constraints satisfied

**Skills required:**
- Pattern recognition
- Color cycle tracking
- Constraint propagation

---

## Gap Analysis: Our Training vs ARC-AGI-3

### What We Trained
| Aspect | SimpleARCEnv |
|--------|--------------|
| Grid size | 5x5 (small) |
| Actions | Pick color for current cell |
| Rewards | Dense (+1 per correct cell) |
| Task | Single transform (copy, reflect) |
| State | Fully observable |
| Planning | None (greedy works) |

### What ARC-AGI-3 Requires
| Aspect | ARC-AGI-3 |
|--------|-----------|
| Grid size | 64x64 |
| Actions | Navigate/Click with coordinates |
| Rewards | Sparse (level complete only) |
| Task | Discover rules, solve puzzles |
| State | Partially hidden rules |
| Planning | Multi-step required |

### Key Gaps

1. **Action space mismatch**
   - Ours: Color selection (4 colors)
   - ARC: Navigation + click coordinates

2. **Reward density**
   - Ours: Every step
   - ARC: Only on level complete (sparse)

3. **Rule discovery**
   - Ours: Rules are known (copy input)
   - ARC: Rules must be discovered through interaction

4. **Scale**
   - Ours: 5x5 = 25 cells
   - ARC: 64x64 = 4096 pixels

5. **Multi-level structure**
   - Ours: Single episode
   - ARC: 6-7 levels with increasing difficulty

---

## Recommended Architecture Adaptations

### 1. Encoder
- Handle 64x64 inputs (current encoder needs modification)
- Extract meaningful features (objects, patterns, relationships)
- Track state changes across steps

### 2. Policy
- Support variable action spaces (navigation vs click)
- Coordinate prediction for click actions
- Sequence modeling for multi-step plans

### 3. World Model
- Learn game dynamics from interaction
- Predict effects of actions
- Enable planning/imagination

### 4. Meta-Learning
- Quickly adapt to new game mechanics
- Learn to discover rules
- Transfer knowledge across games

---

## Proposed Approach

### Phase 1: Basic Integration
- Adapt encoder for 64x64 inputs
- Create action adapter for each game type
- Test random/heuristic baselines

### Phase 2: Rule Discovery
- Implement exploration strategies
- Track state changes to infer rules
- Build rule hypotheses

### Phase 3: Planning
- Use world model for look-ahead
- Implement search/MCTS for action selection
- Budget-aware planning

### Phase 4: Meta-Learning
- Train across game variations
- Learn to adapt quickly to new mechanics
- Generalize patterns across games

---

## Observations from Game Source Code

The games use:
- **Sprites** with properties (position, color, rotation, tags)
- **Collision detection** for interactions
- **State machines** for game logic
- **Resource counters** (lives, budget, moves)
- **Win/lose conditions** based on constraint satisfaction

This suggests our agent needs:
- Object-level understanding (not just pixels)
- Interaction modeling (what happens when X meets Y)
- Goal inference (what are the win conditions?)
- Resource planning (how to use limited actions efficiently)
