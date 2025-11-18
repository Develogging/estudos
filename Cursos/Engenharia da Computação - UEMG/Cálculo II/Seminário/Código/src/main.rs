use bevy::prelude::*;

// --- CONSTANTES DE SIMULAÇÃO ---

// Comprimentos dos segmentos do braço
const LEN1: f32 = 100.0;
const LEN2: f32 = 80.0;
// Taxa de aprendizado para o Gradient Descent.
// Um valor pequeno evita oscilações.
const LEARNING_RATE: f32 = 0.0001;
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

/// Sistema Bevy que implementa o "exemplo e resolver" [cite: Orientações].
/// Este é o núcleo do seu trabalho: Otimização por Gradient Descent.
fn ik_solver_system(mut arm: ResMut<ArmState>, target: Res<Target>) {
    // --- CONEXÃO COM O CÁLCULO ---
    // Este sistema implementa a Seção 3.3 da sua Metodologia [cite: artigo_ik.md].

    // Parâmetros do braço
    let l1 = LEN1;
    let l2 = LEN2;
    let max_reach = l1 + l2;
    let t1 = arm.angles.x; // theta1 (ombro)
    let t2 = arm.angles.y; // theta2 (cotovelo)

    // --- PASSO 1: Cinemática Direta (FK) ---
    // Calcula a posição atual da ponta (efetuador final) `s`
    // Posição da Junta 1 (cotovelo)
    let j1 = Vec2::new(l1 * t1.cos(), l1 * t1.sin());
    // Posição da Junta 2 (ponta)
    let s = j1 + Vec2::new(l2 * (t1 + t2).cos(), l2 * (t1 + t2).sin());

    // --- CORREÇÃO DO "DEBATE" (JITTER) ---
    // Verifique a distância até o alvo.
    let mut target_pos = target.pos;
    let distance_to_target = target_pos.length();

    // Se o alvo estiver fora do alcance máximo...
    if distance_to_target > max_reach {
        // ... "prenda" (clamp) o alvo na borda do círculo de alcance máximo.
        // Isso dá ao otimizador um alvo alcançável e previne a instabilidade.
        target_pos = target_pos.normalize() * max_reach;
    }

    // --- PASSO 2: Calcular a Função de Custo (Erro) ---
    // O erro é a diferença entre o alvo `t` e a posição atual `s`
    let e = target_pos - s;
    let error_distance_sq = e.length_squared();

    // Se estivermos perto o suficiente, pare a otimização.
    if error_distance_sq < STOP_THRESHOLD.powi(2) {
        return;
    }

    // --- PASSO 3: Calcular o Jacobiano (J) ---
    // J é a matriz de derivadas parciais [ds/d(theta)]
    // J = | dx/dt1  dx/dt2 |
    //     | dy/dt1  dy/dt2 |

    // s.x = l1*cos(t1) + l2*cos(t1+t2)
    // s.y = l1*sin(t1) + l2*sin(t1+t2)

    // Derivadas parciais (as 4 equações de cálculo)
    let j11 = -l1 * t1.sin() - l2 * (t1 + t2).sin(); // dx/dt1
    let j12 = -l2 * (t1 + t2).sin(); // dx/dt2
    let j21 = l1 * t1.cos() + l2 * (t1 + t2).cos(); // dy/dt1
    let j22 = l2 * (t1 + t2).cos(); // dy/dt2

    // --- PASSO 4: Otimização (Gradient Descent) ---
    // Aplicamos o algoritmo derivado na Metodologia:
    // delta_theta = alpha * J^T * e

    // Calcular J^T * e
    // J^T = | j11  j21 |
    //       | j12  j22 |
    // (J^T * e)_1 = j11*e.x + j21*e.y
    // (J^T * e)_2 = j12*e.x + j22*e.y
    let delta_t1 = j11 * e.x + j21 * e.y;
    let delta_t2 = j12 * e.x + j22 * e.y;

    // --- PASSO 5: Atualizar os Ângulos ---
    // Aplica a taxa de aprendizado
    let mut dt1 = LEARNING_RATE * delta_t1;
    let mut dt2 = LEARNING_RATE * delta_t2;

    // --- NOVA CORREÇÃO: "AMORTECIMENTO" (DAMPING) ---
    // Limita a mudança máxima por frame para evitar "debater" (jitter)
    // Este é um "amortecimento" manual que previne instabilidade
    // perto de singularidades (quando o braço está totalmente esticado).
    // Isto simula o efeito de algoritmos mais complexos como "Damped Least Squares".
    const MAX_STEP_PER_FRAME: f32 = 0.02; // Limite de ~1.15 graus por frame
    dt1 = dt1.clamp(-MAX_STEP_PER_FRAME, MAX_STEP_PER_FRAME);
    dt2 = dt2.clamp(-MAX_STEP_PER_FRAME, MAX_STEP_PER_FRAME);

    // --- PASSO 5: Atualizar os Ângulos ---
    // Atualiza os ângulos na direção do gradiente negativo
    arm.angles.x += dt1;
    arm.angles.y += dt2;
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
