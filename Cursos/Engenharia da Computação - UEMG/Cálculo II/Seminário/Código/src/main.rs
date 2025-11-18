use bevy::prelude::*;

// --- CONSTANTES DE SIMULAÇÃO ---

// Comprimentos dos segmentos do braço
const LEN1: f32 = 100.0;
const LEN2: f32 = 80.0;
// Taxa de aprendizado para o Gradient Descent.
// Um valor pequeno evita oscilações.
const LEARNING_RATE: f32 = 0.0001;
// Velocidade máxima de rotação (radianos) por frame. ~0.1 rad = 5.7 graus
const MAX_STEP_PER_FRAME: f32 = 0.1;
// Distância de erro mínima para parar a otimização
const STOP_THRESHOLD: f32 = 1.0;

/// Recurso (Resource) Bevy para armazenar o estado do braço (ângulos)
#[derive(Resource)]
struct ArmState {
    // Vec2(theta1, theta2)
    angles: Vec2,
}

impl Default for ArmState {
    fn default() -> Self {
        Self {
            angles: Vec2::new(std::f32::consts::FRAC_PI_2, 0.1), // Começa apontando para cima
        }
    }
}

/// Recurso Bevy para armazenar a posição do alvo (mouse)
#[derive(Resource, Default)]
struct Target {
    pos: Vec2,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .init_resource::<ArmState>()
        .init_resource::<Target>()
        .add_systems(Startup, setup_camera)
        .add_systems(
            Update,
            (
                update_target_position, // Atualiza a posição do alvo (mouse)
                ik_solver_system,       // Executa o algoritmo de otimização
                draw_arm_system,        // Desenha o braço e o alvo
            )
                .chain(), // Garante a ordem de execução
        )
        .run();
}

/// Sistema Bevy para iniciar a câmera 2D
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// Sistema Bevy para ler a posição do mouse e atualizar o recurso Target
fn update_target_position(
    mut target: ResMut<Target>,
    windows: Query<&Window>,
    q_camera: Query<(&Camera, &GlobalTransform)>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Ok((camera, camera_transform)) = q_camera.single() else {
        return;
    };

    // Converte a posição do cursor na tela para a posição no "mundo" 2D
    if let Some(world_position) = window
        .cursor_position()
        .map(|cursor| camera.viewport_to_world(camera_transform, cursor))
        .map(|ray| ray.unwrap().origin.truncate())
    {
        target.pos = world_position;
    }
}

/// O Solver Híbrido: Analítico (Longe) + Gradient Descent (Perto)
fn ik_solver_system(mut arm: ResMut<ArmState>, target: Res<Target>) {
    let l1 = LEN1;
    let l2 = LEN2;
    let max_reach = l1 + l2;

    // --- SUB-STEPPING ---
    // Dividimos o frame em 10 passos menores.
    // Isso aumenta drasticamente a estabilidade e precisão da simulação.
    const SOLVER_STEPS: usize = 10;

    // Ajustamos as constantes para o sub-step
    let sub_lr = LEARNING_RATE / SOLVER_STEPS as f32;
    let sub_max_step = MAX_STEP_PER_FRAME / SOLVER_STEPS as f32;

    for _ in 0..SOLVER_STEPS {
        let t1 = arm.angles.x;
        let t2 = arm.angles.y;

        // FK (Cinemática Direta)
        let j1 = Vec2::new(l1 * t1.cos(), l1 * t1.sin());
        let s = j1 + Vec2::new(l2 * (t1 + t2).cos(), l2 * (t1 + t2).sin());

        let dist_to_target = target.pos.length();

        // --- CASO 1: ALVO FORA DE ALCANCE (Solução Analítica) ---
        // Resolve o problema de "não estender" e o "jitter na ponta".
        if dist_to_target > max_reach {
            // Ângulo global para o alvo
            let target_angle = target.pos.y.atan2(target.pos.x);

            // 1. Queremos que o Ombro (t1) aponte para o alvo
            let mut diff_t1 = target_angle - t1;
            // Garante que giramos pelo lado mais curto (-PI a +PI)
            diff_t1 = (diff_t1 + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;

            // 2. Queremos que o Cotovelo (t2) seja 0 (braço reto)
            let diff_t2 = 0.0 - t2;

            // Aplicamos a mudança suavemente (sem teleporte)
            arm.angles.x += diff_t1.clamp(-sub_max_step, sub_max_step);
            arm.angles.y += diff_t2.clamp(-sub_max_step, sub_max_step);
        }
        // --- CASO 2: ALVO DENTRO DE ALCANCE (Gradient Descent) ---
        else {
            let e = target.pos - s;
            if e.length_squared() < STOP_THRESHOLD.powi(2) {
                continue;
            }

            // Jacobiano Transposto
            let j11 = -l1 * t1.sin() - l2 * (t1 + t2).sin();
            let j12 = -l2 * (t1 + t2).sin();
            let j21 = l1 * t1.cos() + l2 * (t1 + t2).cos();
            let j22 = l2 * (t1 + t2).cos();

            let d_t1 = j11 * e.x + j21 * e.y;
            let d_t2 = j12 * e.x + j22 * e.y;

            // Atualização com Amortecimento
            arm.angles.x += (sub_lr * d_t1).clamp(-sub_max_step, sub_max_step);
            arm.angles.y += (sub_lr * d_t2).clamp(-sub_max_step, sub_max_step);
        }
    }
}

/// Sistema Bevy para desenhar o estado atual do braço e o alvo
fn draw_arm_system(mut gizmos: Gizmos, arm: Res<ArmState>, target: Res<Target>) {
    let t1 = arm.angles.x;
    let t2 = arm.angles.y;

    // Posições das juntas (calculado via FK novamente)
    let base = Vec2::ZERO;
    let j1 = base + Vec2::new(LEN1 * t1.cos(), LEN1 * t1.sin());
    let j2 = j1 + Vec2::new(LEN2 * (t1 + t2).cos(), LEN2 * (t1 + t2).sin());

    // Desenha o braço
    gizmos.line_2d(base, j1, Color::srgba(25.0, 32.0, 52.0, 0.6));
    gizmos.circle_2d(j1, 5.0, Color::WHITE);
    gizmos.line_2d(j1, j2, Color::srgba(25.0, 32.0, 52.0, 0.6));
    gizmos.circle_2d(j2, 5.0, Color::srgba(255.0, 0.0, 0.0, 1.0)); // Ponta (efetuador final)

    // Desenha o alvo (mouse)
    gizmos.circle_2d(target.pos, 10.0, Color::srgba(0.0, 255.0, 0.0, 1.0));
}
