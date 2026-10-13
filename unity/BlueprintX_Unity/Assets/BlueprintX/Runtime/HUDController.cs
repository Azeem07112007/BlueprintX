using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

// =======================================================
//  HUDController.cs  —  BlueprintX Premium HUD
//  Shows:
//   • Animated loading screen while building generates
//   • Room name + area overlay once loaded
//   • Controls cheat-sheet
//   • Emergency evacuation mode
// =======================================================

namespace BlueprintX.Runtime
{
    public class HUDController : MonoBehaviour
    {
        [Header("References")]
        public BuildingLoader  buildingLoader;
        public PlayerInteraction playerInteraction;

        [Header("State")]
        public bool   showHUD           = true;
        public bool   emergencyMode     = false;
        public bool   isBuildingLoading = false;
        public float  loadingProgress   = 0f;
        public string currentRoomName   = "Initialising…";
        public float  currentRoomArea   = 0f;

        // internal GUI styles — created once
        private GUIStyle sTitleBig;
        private GUIStyle sTitleSmall;
        private GUIStyle sRoom;
        private GUIStyle sBody;
        private GUIStyle sPrompt;
        private GUIStyle sEmergency;
        private GUIStyle sLoadBar;
        private GUIStyle sLoadBarBg;
        private bool stylesInitialised = false;

        private float loadingAnimTime = 0f;

        private void Update()
        {
            loadingAnimTime += Time.deltaTime;

            bool toggleEmergency = false;
#if ENABLE_INPUT_SYSTEM
            if (Keyboard.current != null)
                toggleEmergency = Keyboard.current.fKey.wasPressedThisFrame;
#else
            toggleEmergency = Input.GetKeyDown(KeyCode.F);
#endif
            if (toggleEmergency)
                emergencyMode = !emergencyMode;
        }

        private void OnGUI()
        {
            if (!showHUD) return;
            InitStyles();

            // ─────────────────────────────────────────────────
            //  LOADING SCREEN  (only while building is loading)
            // ─────────────────────────────────────────────────
            if (isBuildingLoading)
            {
                DrawLoadingScreen();
                return;   // don't draw the normal HUD during load
            }

            // ─────────────────────────────────────────────────
            //  MAIN GAME HUD
            // ─────────────────────────────────────────────────

            // Crosshair
            float cx = Screen.width  * 0.5f;
            float cy = Screen.height * 0.5f;
            DrawCrosshair(cx, cy);

            // Hover prompt
            if (playerInteraction != null && !string.IsNullOrEmpty(playerInteraction.hoverPrompt))
                GUI.Label(new Rect(cx - 150f, cy + 30f, 300f, 28f), playerInteraction.hoverPrompt, sPrompt);

            // Top-left info panel
            float panelW = 320f, panelH = 220f;
            DrawPanel(new Rect(20, 20, panelW, panelH));

            GUILayout.BeginArea(new Rect(30, 30, panelW - 20, panelH - 20));
            GUILayout.Label("BLUEPRINT-X  3D", sTitleBig);
            GUILayout.Space(2);

            string roomLabel = !string.IsNullOrEmpty(currentRoomName) ? currentRoomName : "—";
            GUILayout.Label($"📍  {roomLabel}", sRoom);

            if (currentRoomArea > 0f)
                GUILayout.Label($"Area: {currentRoomArea:F1} m²", sBody);

            if (buildingLoader?.buildingData != null)
            {
                int rooms = buildingLoader.buildingData.rooms?.Length ?? 0;
                int doors = buildingLoader.buildingData.doors?.Length ?? 0;
                int walls = buildingLoader.buildingData.walls?.Length ?? 0;
                GUILayout.Label($"Rooms: {rooms}   Doors: {doors}   Walls: {walls}", sBody);
                GUILayout.Label($"Scale: {buildingLoader.modelScaleMultiplier:F1}×   " +
                                $"Floor h: {buildingLoader.wallHeight:F1} m", sBody);
            }

            string ceilStr = (buildingLoader != null && buildingLoader.showCeiling)
                ? "Ceiling ON  ·  [C] cutaway"
                : "Ceiling OFF  ·  [C] restore";
            GUILayout.Label(ceilStr, sBody);

            GUILayout.Space(4);
            GUILayout.Label("WASD · Mouse look · E door · C ceiling · F evac", sBody);

            if (emergencyMode)
            {
                GUILayout.Space(4);
                GUILayout.Label("⚠️  EMERGENCY EVACUATION ACTIVE", sEmergency);
                GUILayout.Label("Routing to nearest exit…", sBody);
            }

            GUILayout.EndArea();
        }

        // ─────────────────────────────────────────────────────
        //  LOADING SCREEN RENDERER
        // ─────────────────────────────────────────────────────
        private void DrawLoadingScreen()
        {
            int W = Screen.width, H = Screen.height;

            // Full-screen dark overlay
            GUI.color = new Color(0.06f, 0.06f, 0.10f, 0.95f);
            GUI.DrawTexture(new Rect(0, 0, W, H), Texture2D.whiteTexture);
            GUI.color = Color.white;

            // Brand title
            GUI.Label(new Rect(W * 0.5f - 220f, H * 0.28f, 440f, 60f), "BLUEPRINT-X  3D", sTitleBig);

            // Sub-tagline
            GUI.Label(new Rect(W * 0.5f - 200f, H * 0.38f, 400f, 30f),
                      "GPU-Accelerated Architectural Renderer", sTitleSmall);

            // Status message
            GUI.Label(new Rect(W * 0.5f - 250f, H * 0.50f, 500f, 30f), currentRoomName, sBody);

            // Progress bar background
            Rect barBg = new Rect(W * 0.5f - 200f, H * 0.57f, 400f, 18f);
            GUI.color = new Color(0.20f, 0.22f, 0.28f, 1f);
            GUI.DrawTexture(barBg, Texture2D.whiteTexture);

            // Progress bar fill — animated glow
            float pulse = 0.85f + 0.15f * Mathf.Sin(loadingAnimTime * 4f);
            Rect barFill = new Rect(barBg.x + 2, barBg.y + 2,
                                    (barBg.width - 4) * Mathf.Clamp01(loadingProgress),
                                    barBg.height - 4);
            GUI.color = new Color(0.25f * pulse, 0.65f * pulse, 1.0f * pulse, 1f);
            GUI.DrawTexture(barFill, Texture2D.whiteTexture);

            GUI.color = Color.white;

            // Percentage text
            string pct = $"{Mathf.RoundToInt(loadingProgress * 100f)}%";
            GUI.Label(new Rect(W * 0.5f - 30f, H * 0.60f, 60f, 24f), pct, sBody);

            // Animated spinner dots
            int dotCount = 3;
            int activeDot = Mathf.FloorToInt(loadingAnimTime * 2f) % dotCount;
            for (int i = 0; i < dotCount; i++)
            {
                float alpha = (i == activeDot) ? 1f : 0.3f;
                GUI.color = new Color(0.4f, 0.75f, 1f, alpha);
                GUI.DrawTexture(new Rect(W * 0.5f - 18f + i * 18f, H * 0.67f, 10f, 10f), Texture2D.whiteTexture);
            }
            GUI.color = Color.white;
        }

        // ─────────────────────────────────────────────────────
        //  HELPERS
        // ─────────────────────────────────────────────────────
        private void DrawCrosshair(float cx, float cy)
        {
            GUI.color = new Color(1f, 1f, 1f, 0.80f);
            GUI.DrawTexture(new Rect(cx - 8f, cy - 1f, 16f, 2f), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(cx - 1f, cy - 8f, 2f, 16f), Texture2D.whiteTexture);
            GUI.color = Color.white;
        }

        private void DrawPanel(Rect r)
        {
            GUI.color = new Color(0.06f, 0.07f, 0.12f, 0.82f);
            GUI.DrawTexture(r, Texture2D.whiteTexture);
            GUI.color = new Color(0.28f, 0.55f, 1.0f, 0.55f);
            GUI.DrawTexture(new Rect(r.x, r.y, r.width, 2f), Texture2D.whiteTexture);   // top accent line
            GUI.color = Color.white;
        }

        private void InitStyles()
        {
            if (stylesInitialised) return;

            sTitleBig = new GUIStyle(GUI.skin.label)
            {
                fontSize  = 22,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                normal    = { textColor = new Color(0.95f, 0.85f, 0.30f) }
            };

            sTitleSmall = new GUIStyle(GUI.skin.label)
            {
                fontSize  = 13,
                fontStyle = FontStyle.Normal,
                alignment = TextAnchor.MiddleCenter,
                normal    = { textColor = new Color(0.70f, 0.80f, 1.0f) }
            };

            sRoom = new GUIStyle(GUI.skin.label)
            {
                fontSize  = 15,
                fontStyle = FontStyle.Bold,
                normal    = { textColor = Color.white }
            };

            sBody = new GUIStyle(GUI.skin.label)
            {
                fontSize  = 11,
                normal    = { textColor = new Color(0.80f, 0.83f, 0.88f) }
            };

            sPrompt = new GUIStyle(GUI.skin.label)
            {
                fontSize  = 14,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                normal    = { textColor = Color.yellow }
            };

            sEmergency = new GUIStyle(GUI.skin.label)
            {
                fontSize  = 13,
                fontStyle = FontStyle.Bold,
                normal    = { textColor = new Color(1f, 0.35f, 0.25f) }
            };

            sLoadBar   = new GUIStyle(GUI.skin.box);
            sLoadBarBg = new GUIStyle(GUI.skin.box);

            stylesInitialised = true;
        }
    }
}
