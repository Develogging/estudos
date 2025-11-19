use bevy::prelude::*;

// --- CONSTANTES DE CONFIGURAÇÃO E OTIMIZAÇÃO ---

/// Comprimento do segmento umeral (L1).
const LEN1: f32 = 100.0;
/// Comprimento do segmento ulnar (L2).
const LEN2: f32 = 80.0;

/// Taxa de aprendizado (learning rate - alpha).
/// Define o tamanho do passo dado na direção do gradiente negativo.
/// Valores muito altos causam oscilação (overshooting); muito baixos causam convergência lenta.
const LEARNING_RATE: f32 = 0.0001;

/// Limite de velocidade angular por quadro (clamping).
/// Atua como um amortecedor não-linear, prevenindo que atualizações bruscas
/// do gradiente desestabilizem a simulação visual.
/// ~0.1 rad equivale a aprox. 5.7 graus.
const MAX_STEP_PER_FRAME: f32 = 0.1;

/// Limiar de erro para parada (critério de convergência).
/// Se a distância entre o efetuador e o alvo for menor que isso, a otimização cessa
/// para economizar processamento.
const STOP_THRESHOLD: f32 = 1.0;

/// Resource que armazena o vetor de estado do sistema (espaço das juntas).
/// Representa o vetor theta = [theta1, theta2]^T.
#[derive(Resource)]
struct ArmState {
    /// x: theta1 (ângulo da base/ombro)
    /// y: theta2 (ângulo relativo do cotovelo)
    angles: Vec2,
}

impl Default for ArmState {
    fn default() -> Self {
        Self {
            // Configuração inicial arbitrária (braço levemente flexionado para cima)
            angles: Vec2::new(std::f32::consts::FRAC_PI_2, 0.1),
        }
    }
}

/// Recurso que armazena a posição do alvo (espaço cartesiano).
/// Representa o vetor t (target).
#[derive(Resource, Default)]
struct Target {
    pos: Vec2,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        // Inicialização dos resources (estado do robô e alvo)
        .init_resource::<ArmState>()
        .init_resource::<Target>()
        .add_systems(Startup, setup_camera)
        .add_systems(
            Update,
            (
                // 1. Captura a intenção do usuário (input)
                update_target_position,
                // 2. Resolve a matemática (IK via Otimização)
                ik_solver_system,
                // 3. Renderiza o resultado visual
                draw_arm_system,
            )
                .chain(), // Garante a execução sequencial
        )
        .run();
}

/// Inicializa a câmera ortográfica 2D para visualização.
fn setup_camera(mut commands: Commands) {
    commands.spawn(Camera2d);
}

/// Sistema de input: mapeia coordenadas de tela para coordenadas de mundo.
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

    // Raycasting: projeta a posição do mouse da viewport para o plano cartesiano.
    if let Some(world_position) = window
        .cursor_position()
        .map(|cursor| camera.viewport_to_world(camera_transform, cursor))
        .map(|ray| ray.unwrap().origin.truncate())
    {
        target.pos = world_position;
    }
}

/// Solução híbrida da cinemática inversa.
/// Combina uma solução analítica para alvos distantes com gradient descent
/// para alvos alcançáveis, garantindo robustez e estabilidade.
fn ik_solver_system(mut arm: ResMut<ArmState>, target: Res<Target>) {
    let l1 = LEN1;
    let l2 = LEN2;
    // Alcance máximo teórico do manipulador (singularidade de borda)
    let max_reach = l1 + l2;

    // --- SUB-STEPPING ---
    // Dividimos o "delta time" do frame em passos menores de integração.
    // Isso lineariza melhor o problema não-linear, permitindo uma convergência mais suave e estável.
    const SOLVER_STEPS: usize = 10;

    // Ajuste proporcional dos parâmetros para manter a consistência física independente dos passos.
    let sub_lr = LEARNING_RATE / SOLVER_STEPS as f32;
    let sub_max_step = MAX_STEP_PER_FRAME / SOLVER_STEPS as f32;

    for _ in 0..SOLVER_STEPS {
        // Leitura do estado atual (theta)
        let t1 = arm.angles.x;
        let t2 = arm.angles.y;

        // --- CINEMÁTICA DIRETA (Forward Kinematics - FK) ---
        // Calcula a posição atual do efetuador s(theta) baseada nos ângulos atuais.
        // s_x = l1*cos(t1) + l2*cos(t1+t2)
        // s_y = l1*sen(t1) + l2*sen(t1+t2)

        // Posição do cotovelo (j1)
        let j1 = Vec2::new(l1 * t1.cos(), l1 * t1.sin());
        // Posição do efetuador final (s)
        let s = j1 + Vec2::new(l2 * (t1 + t2).cos(), l2 * (t1 + t2).sin());

        let dist_to_target = target.pos.length();

        // --- ESTRATÉGIA HÍBRIDA ---

        // CASO 1: ALVO FORA DE ALCANCE
        // Se o alvo está impossivelmente longe, o gradiente descendente tentaria esticar
        // o braço infinitamente, causando oscilação (jitter) e desperdício de CPU.
        // Solução: Alinhar geometricamente o braço na direção do alvo.
        if dist_to_target > max_reach {
            // Vetor direção do alvo
            let target_angle = target.pos.y.atan2(target.pos.x);

            // 1. O ombro (t1) deve apontar diretamente para o alvo
            let mut diff_t1 = target_angle - t1;

            // Normalização angular: garante que a rotação ocorra pelo menor arco (-PI a +PI)
            diff_t1 = (diff_t1 + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU)
                - std::f32::consts::PI;

            // 2. O cotovelo (t2) deve ser 0 (braço completamente esticado)
            let diff_t2 = 0.0 - t2;

            // Aplica a correção com limitação de velocidade (clamping)
            arm.angles.x += diff_t1.clamp(-sub_max_step, sub_max_step);
            arm.angles.y += diff_t2.clamp(-sub_max_step, sub_max_step);
        }
        // CASO 2: ALVO DENTRO DE ALCANCE (Otimização via gradiente)
        else {
            // Vetor de erro: e = t - s(theta)
            let e = target.pos - s;

            // Critério de parada antecipada (se já estivermos perto o suficiente)
            if e.length_squared() < STOP_THRESHOLD.powi(2) {
                continue;
            }

            // --- CÁLCULO DA MATRIZ JACOBIANA (J) ---
            // A jacobiana relaciona as velocidades articulares às velocidades cartesianas.
            // J = [ dx/dt1  dx/dt2 ]
            //     [ dy/dt1  dy/dt2 ]

            // Derivadas parciais de x e y em relação a theta1 e theta2
            let j11 = -l1 * t1.sin() - l2 * (t1 + t2).sin(); // dx/dt1
            let j12 = -l2 * (t1 + t2).sin(); // dx/dt2
            let j21 = l1 * t1.cos() + l2 * (t1 + t2).cos(); // dy/dt1
            let j22 = l2 * (t1 + t2).cos(); // dy/dt2

            // --- MÉTODO DO JACOBIANO TRANSPOSTO ---
            // Em vez de calcular a inversa (J^-1), usamos a transposta (J^T) como uma
            // aproximação válida da direção do gradiente para minimizar o erro.
            // delta_theta = alpha * J^T * erro

            let d_t1 = j11 * e.x + j21 * e.y; // Linha 1 da Transposta multiplicada pelo erro
            let d_t2 = j12 * e.x + j22 * e.y; // Linha 2 da Transposta multiplicada pelo erro

            // Atualização dos ângulos (theta_novo = theta_antigo + delta_theta)
            // Inclui "clamping" para simular amortecimento e garantir estabilidade
            arm.angles.x += (sub_lr * d_t1).clamp(-sub_max_step, sub_max_step);
            arm.angles.y += (sub_lr * d_t2).clamp(-sub_max_step, sub_max_step);
        }
    }
}

/// Sistema de visualização (Gizmos).
/// Recalcula a FK apenas para desenhar o estado atual na tela.
fn draw_arm_system(mut gizmos: Gizmos, arm: Res<ArmState>, target: Res<Target>) {
    let t1 = arm.angles.x;
    let t2 = arm.angles.y;

    // Recálculo da FK para visualização
    let base = Vec2::ZERO;
    let j1 = base + Vec2::new(LEN1 * t1.cos(), LEN1 * t1.sin());
    let j2 = j1 + Vec2::new(LEN2 * (t1 + t2).cos(), LEN2 * (t1 + t2).sin());

    // Desenho dos segmentos (cinza)
    gizmos.line_2d(base, j1, Color::srgba(25.0, 32.0, 52.0, 0.6));
    gizmos.circle_2d(j1, 5.0, Color::WHITE); // Junta 1 (cotovelo)

    gizmos.line_2d(j1, j2, Color::srgba(25.0, 32.0, 52.0, 0.6));
    gizmos.circle_2d(j2, 5.0, Color::srgba(255.0, 0.0, 0.0, 1.0)); // Efetuador final (ponta)

    // Desenho do alvo desejado (verde)
    gizmos.circle_2d(target.pos, 10.0, Color::srgba(0.0, 255.0, 0.0, 1.0));
}
