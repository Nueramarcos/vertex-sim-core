// Vertex - Simulation Core (lib crate)
// Tick-based simulation engine with a component-style entity system.

use std::collections::HashMap;
use std::fmt;

// ── Config & State ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SimConfig {
    pub max_ticks: u64,
    pub time_step: f64,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self { max_ticks: 1_000, time_step: 0.016 }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SimState {
    pub tick: u64,
    pub elapsed: f64,
}

// ── Entity system ─────────────────────────────────────────────────────────────

pub type EntityId = u64;

/// An arbitrary component value stored on an entity.
#[derive(Debug, Clone)]
pub enum Component {
    Position { x: f64, y: f64 },
    Velocity { dx: f64, dy: f64 },
    Health(f64),
    Tag(String),
    Custom(String, f64),   // (key, value) catch-all
}

impl Component {
    pub fn kind(&self) -> &'static str {
        match self {
            Component::Position { .. } => "Position",
            Component::Velocity { .. } => "Velocity",
            Component::Health(_)       => "Health",
            Component::Tag(_)          => "Tag",
            Component::Custom(_, _)    => "Custom",
        }
    }
}

/// Sparse component storage: entity → list of components.
#[derive(Debug, Default)]
pub struct ComponentStore {
    data: HashMap<EntityId, Vec<Component>>,
    next_id: EntityId,
}

impl ComponentStore {
    pub fn spawn(&mut self) -> EntityId {
        let id = self.next_id;
        self.next_id += 1;
        self.data.insert(id, vec![]);
        id
    }

    pub fn despawn(&mut self, id: EntityId) -> bool {
        self.data.remove(&id).is_some()
    }

    pub fn add(&mut self, id: EntityId, component: Component) {
        self.data.entry(id).or_default().push(component);
    }

    pub fn get(&self, id: EntityId) -> Option<&Vec<Component>> {
        self.data.get(&id)
    }

    pub fn get_mut(&mut self, id: EntityId) -> Option<&mut Vec<Component>> {
        self.data.get_mut(&id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntityId, &Vec<Component>)> {
        self.data.iter().map(|(&id, comps)| (id, comps))
    }

    pub fn entity_count(&self) -> usize {
        self.data.len()
    }
}

// ── Built-in systems ──────────────────────────────────────────────────────────

/// Apply velocity components to position components (Euler integration).
pub fn system_movement(store: &mut ComponentStore, dt: f64) {
    // Collect velocity values first (borrow-checker friendly)
    let velocities: Vec<(EntityId, f64, f64)> = store
        .iter()
        .filter_map(|(id, comps)| {
            comps.iter().find_map(|c| {
                if let Component::Velocity { dx, dy } = c {
                    Some((id, *dx, *dy))
                } else {
                    None
                }
            })
        })
        .collect();

    for (id, dx, dy) in velocities {
        if let Some(comps) = store.get_mut(id) {
            for comp in comps.iter_mut() {
                if let Component::Position { x, y } = comp {
                    *x += dx * dt;
                    *y += dy * dt;
                }
            }
        }
    }
}

// ── Simulation ────────────────────────────────────────────────────────────────

pub struct Simulation {
    pub config: SimConfig,
    pub state: SimState,
    pub entities: ComponentStore,
    running: bool,
}

impl Simulation {
    pub fn new(config: SimConfig) -> Self {
        Self {
            config,
            state: SimState::default(),
            entities: ComponentStore::default(),
            running: false,
        }
    }

    pub fn tick(&mut self) -> bool {
        if self.state.tick >= self.config.max_ticks {
            self.running = false;
            return false;
        }
        // Run built-in movement system every tick
        system_movement(&mut self.entities, self.config.time_step);
        self.state.tick += 1;
        self.state.elapsed += self.config.time_step;
        true
    }

    pub fn run<F>(&mut self, mut step_fn: F)
    where
        F: FnMut(&SimState, &ComponentStore) -> bool,
    {
        self.running = true;
        while self.running && self.tick() {
            if !step_fn(&self.state, &self.entities) {
                self.running = false;
            }
        }
    }

    pub fn is_running(&self) -> bool { self.running }

    pub fn reset(&mut self) {
        self.state = SimState::default();
        self.running = false;
    }
}

impl fmt::Display for Simulation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Simulation {{ tick: {}/{}, elapsed: {:.3}s, entities: {} }}",
            self.state.tick,
            self.config.max_ticks,
            self.state.elapsed,
            self.entities.entity_count(),
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sim(max_ticks: u64) -> Simulation {
        Simulation::new(SimConfig { max_ticks, time_step: 1.0 })
    }

    #[test]
    fn test_spawn_and_despawn() {
        let mut store = ComponentStore::default();
        let id = store.spawn();
        assert_eq!(store.entity_count(), 1);
        assert!(store.despawn(id));
        assert_eq!(store.entity_count(), 0);
    }

    #[test]
    fn test_add_and_get_component() {
        let mut store = ComponentStore::default();
        let id = store.spawn();
        store.add(id, Component::Health(100.0));
        let comps = store.get(id).unwrap();
        assert!(matches!(comps[0], Component::Health(h) if h == 100.0));
    }

    #[test]
    fn test_movement_system() {
        let mut store = ComponentStore::default();
        let id = store.spawn();
        store.add(id, Component::Position { x: 0.0, y: 0.0 });
        store.add(id, Component::Velocity { dx: 2.0, dy: -1.0 });
        system_movement(&mut store, 1.0);
        let comps = store.get(id).unwrap();
        let pos = comps.iter().find_map(|c| {
            if let Component::Position { x, y } = c { Some((*x, *y)) } else { None }
        });
        assert_eq!(pos, Some((2.0, -1.0)));
    }

    #[test]
    fn test_sim_tick_moves_entities() {
        let mut sim = make_sim(3);
        let id = sim.entities.spawn();
        sim.entities.add(id, Component::Position { x: 0.0, y: 0.0 });
        sim.entities.add(id, Component::Velocity { dx: 1.0, dy: 0.0 });
        sim.tick();
        let comps = sim.entities.get(id).unwrap();
        let x = comps.iter().find_map(|c| {
            if let Component::Position { x, .. } = c { Some(*x) } else { None }
        });
        assert_eq!(x, Some(1.0));
    }

    #[test]
    fn test_run_callback_receives_state() {
        let mut sim = make_sim(4);
        let mut ticks_seen = 0u64;
        sim.run(|state, _entities| {
            ticks_seen = state.tick;
            true
        });
        assert_eq!(ticks_seen, 4);
    }

    #[test]
    fn test_reset() {
        let mut sim = make_sim(10);
        sim.tick();
        sim.reset();
        assert_eq!(sim.state.tick, 0);
    }

    #[test]
    fn test_default_config_values() {
        let config = SimConfig::default();
        assert_eq!(config.max_ticks, 1_000);
        assert_eq!(config.time_step, 0.016);
    }
}
