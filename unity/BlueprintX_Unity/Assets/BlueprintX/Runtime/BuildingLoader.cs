using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

// ============================================================
//  BuildingLoader.cs  —  BlueprintX  Premium 3D Renderer
//  Builds a fully accurate, GPU-rendered architectural model
//  directly from the JSON floor-plan data.
//  No OBJ file dependency.  All geometry is procedural.
// ============================================================

namespace BlueprintX.Runtime
{
    public class BuildingLoader : MonoBehaviour
    {
        // ── Inspector ─────────────────────────────────────────
        [Header("Files")]
        public string jsonFileName = "building_3d.json";

        [Header("References")]
        public FPSController playerController;
        public HUDController hudController;

        [Header("Building Data (auto-populated)")]
        public BuildingJson buildingData;
        public Transform  generatedBuildingRoot;

        [Header("Scale")]
        [Tooltip("Global scale multiplier. 3.0 gives full-size walking-scale rooms.")]
        public float modelScaleMultiplier = 3.0f;

        [Header("Minimum Wall Dimensions (metres, after scale)")]
        public float minOuterWallThickness    = 0.30f;
        public float minInternalWallThickness = 0.20f;
        public float wallHeight               = 3.2f;   // storey height metres (already in world units)

        [Header("Ceiling")]
        public GameObject ceilingContainer;
        public bool showCeiling = true;

        // ── Private state ──────────────────────────────────────
        private float pxPerM   = 100f;
        private float imgHeight = 700f;

        private Material matOuterWall;
        private Material matInnerWall;
        private Material matFloor;
        private Material matCeiling;
        private Material matDoor;
        private Material matWindowFrame;
        private Material matWindowGlass;
        private Material matGrass;
        private Material matPavement;
        private Material matWalkway;
        private Material matTrunkBark;
        private Material matLeaf;

        private bool isLoading = false;

        // ── Unity entry ────────────────────────────────────────
        private void Start()
        {
            if (generatedBuildingRoot == null)
            {
                var go = GameObject.Find("GeneratedBuilding");
                if (go != null) generatedBuildingRoot = go.transform;
                else            generatedBuildingRoot = new GameObject("GeneratedBuilding").transform;
            }

            StartCoroutine(LoadBuildingAsync());
        }

        // ── Public reload API ──────────────────────────────────
        public void LoadBuilding() => StartCoroutine(LoadBuildingAsync());

        // ══════════════════════════════════════════════════════
        //  ASYNC LOADER  — yields each phase so the game loop
        //  keeps running and the loading screen stays smooth
        // ══════════════════════════════════════════════════════
        private BuildingJson ReadOrGenerateBuildingData()
        {
            string jsonPath = null;
            string[] searchPaths = new string[]
            {
                Path.Combine(Application.streamingAssetsPath, jsonFileName),
                Path.Combine(Application.dataPath, "..", jsonFileName),
                Path.Combine(Application.dataPath, jsonFileName),
                Path.Combine(Directory.GetCurrentDirectory(), jsonFileName),
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, jsonFileName),
                Path.Combine(Application.persistentDataPath, jsonFileName),
                Path.Combine(Application.dataPath, "..", "..", jsonFileName),
                @"D:\BlueprintX\building_3d.json",
                @"D:\ME GA\GAmes\to be named\BlueprintX_Unity\Assets\StreamingAssets\building_3d.json"
            };

            foreach (var p in searchPaths)
            {
                try
                {
                    if (!string.IsNullOrEmpty(p) && File.Exists(p))
                    {
                        jsonPath = p;
                        break;
                    }
                }
                catch { }
            }

            BuildingJson data = null;
            if (jsonPath != null)
            {
                try
                {
                    string jsonText = File.ReadAllText(jsonPath);
                    data = JsonUtility.FromJson<BuildingJson>(jsonText);
                }
                catch (Exception ex)
                {
                    Debug.LogWarning($"[BuildingLoader] Exception parsing JSON at {jsonPath}: {ex.Message}");
                }
            }

            if (data == null || data.building == null || data.walls == null || data.walls.Length == 0)
            {
                Debug.LogWarning("[BuildingLoader] No valid building_3d.json found. Generating fallback procedural model...");
                data = GenerateDefaultBuildingJson();
            }

            if (data.building == null)
                data.building = new BuildingParametersJson { width_m = 12f, depth_m = 10f };

            return data;
        }

        private Shader FindShader(params string[] names)
        {
            if (names != null)
            {
                foreach (var name in names)
                {
                    if (string.IsNullOrEmpty(name)) continue;
                    Shader s = Shader.Find(name);
                    if (s != null) return s;
                }
            }
            return Shader.Find("Sprites/Default") ??
                   Shader.Find("Unlit/Color") ??
                   Shader.Find("Legacy Shaders/Diffuse") ??
                   Shader.Find("Hidden/InternalErrorShader");
        }

        private void GetBuildingBounds(out Vector3 center, out float width, out float depth)
        {
            float minX = float.MaxValue, maxX = float.MinValue;
            float minZ = float.MaxValue, maxZ = float.MinValue;
            bool foundAny = false;

            if (buildingData?.walls != null && buildingData.walls.Length > 0)
            {
                foreach (var w in buildingData.walls)
                {
                    float x1 = (w.start[0] / pxPerM) * modelScaleMultiplier;
                    float z1 = ((imgHeight - w.start[1]) / pxPerM) * modelScaleMultiplier;
                    float x2 = (w.end[0] / pxPerM) * modelScaleMultiplier;
                    float z2 = ((imgHeight - w.end[1]) / pxPerM) * modelScaleMultiplier;

                    minX = Mathf.Min(minX, Mathf.Min(x1, x2));
                    maxX = Mathf.Max(maxX, Mathf.Max(x1, x2));
                    minZ = Mathf.Min(minZ, Mathf.Min(z1, z2));
                    maxZ = Mathf.Max(maxZ, Mathf.Max(z1, z2));
                    foundAny = true;
                }
            }

            if (buildingData?.rooms != null)
            {
                foreach (var r in buildingData.rooms)
                {
                    if (r.polygon_flat != null)
                    {
                        for (int i = 0; i < r.polygon_flat.Length / 2; i++)
                        {
                            float rx = (r.polygon_flat[i * 2] / pxPerM) * modelScaleMultiplier;
                            float rz = ((imgHeight - r.polygon_flat[i * 2 + 1]) / pxPerM) * modelScaleMultiplier;
                            minX = Mathf.Min(minX, rx);
                            maxX = Mathf.Max(maxX, rx);
                            minZ = Mathf.Min(minZ, rz);
                            maxZ = Mathf.Max(maxZ, rz);
                            foundAny = true;
                        }
                    }
                }
            }

            if (!foundAny)
            {
                width = Mathf.Max(6f, (buildingData?.building != null ? buildingData.building.width_m : 10f) * modelScaleMultiplier);
                depth = Mathf.Max(6f, (buildingData?.building != null ? buildingData.building.depth_m : 10f) * modelScaleMultiplier);
                center = new Vector3(width * 0.5f, 0f, depth * 0.5f);
                return;
            }

            width = Mathf.Max(6f, maxX - minX);
            depth = Mathf.Max(6f, maxZ - minZ);
            center = new Vector3((minX + maxX) * 0.5f, 0f, (minZ + maxZ) * 0.5f);
        }

        private void SnapWallEndpoints()
        {
            if (buildingData?.walls == null || buildingData.walls.Length == 0) return;

            float snapThresholdPx = 0.15f * pxPerM; // 15cm conservative corner snap threshold

            List<float[]> points = new List<float[]>();
            foreach (var w in buildingData.walls)
            {
                if (w.start != null && w.start.Length >= 2) points.Add(w.start);
                if (w.end != null && w.end.Length >= 2) points.Add(w.end);
            }

            for (int i = 0; i < points.Count; i++)
            {
                for (int j = i + 1; j < points.Count; j++)
                {
                    // Never snap endpoints of the same wall segment together
                    if ((i / 2) == (j / 2)) continue;

                    float dx = points[i][0] - points[j][0];
                    float dy = points[i][1] - points[j][1];
                    float distSq = dx * dx + dy * dy;
                    if (distSq < snapThresholdPx * snapThresholdPx)
                    {
                        points[j][0] = points[i][0];
                        points[j][1] = points[i][1];
                    }
                }
            }
        }

        private IEnumerator LoadBuildingAsync()
        {
            if (isLoading) yield break;
            isLoading = true;

            Exception phaseException = null;

            void RunPhase(Action action)
            {
                if (phaseException != null) return;
                try
                {
                    action();
                }
                catch (Exception ex)
                {
                    phaseException = ex;
                }
            }

            // ── Phase 0: Show Loading Screen ──────────────────
            ShowLoadingHUD("Initialising GPU renderer…", 0f);
            yield return null;

            // ── Phase 1: Read JSON ────────────────────────────
            ShowLoadingHUD("Reading floor-plan data…", 0.05f);
            yield return null;

            RunPhase(() =>
            {
                buildingData = ReadOrGenerateBuildingData();
                pxPerM    = (buildingData.coordinate_system != null && buildingData.coordinate_system.px_per_m > 0f)
                            ? buildingData.coordinate_system.px_per_m : 100f;
                imgHeight = (buildingData.building != null)
                            ? buildingData.building.depth_m * pxPerM : 700f;

                Debug.Log($"[BuildingLoader] Loaded: {buildingData.walls?.Length ?? 0} walls, " +
                          $"{buildingData.rooms?.Length ?? 0} rooms, pxPerM={pxPerM}, imgH={imgHeight}");
            });

            Vector3 bCenter;
            float bW, bD;
            GetBuildingBounds(out bCenter, out bW, out bD);

            // ── Phase 2: Prepare scene & materials ────────────
            ShowLoadingHUD("Compiling GPU materials…", 0.10f);
            yield return null;

            RunPhase(() =>
            {
                ClearBuilding();
                BuildMaterials();
                ConfigureSceneLighting();
                ConfigureCamera();
            });

            // ── Phase 3: Outdoor environment ──────────────────
            ShowLoadingHUD("Generating outdoor terrain…", 0.15f);
            yield return null;

            RunPhase(() => BuildOutdoorEnvironment(bCenter, bW, bD));
            yield return null;

            // ── Phase 4: Floor slabs per room ─────────────────
            ShowLoadingHUD("Rendering architectural floors…", 0.25f);
            yield return null;

            RunPhase(() => BuildFloors(bCenter, bW, bD));
            yield return null;

            // ── Phase 5: Walls ────────────────────────────────
            ShowLoadingHUD("Constructing structural walls…", 0.40f);
            yield return null;

            if (buildingData?.walls != null && buildingData.walls.Length > 0 && phaseException == null)
            {
                SnapWallEndpoints();
                int total = buildingData.walls.Length;
                for (int i = 0; i < total; i++)
                {
                    int wallIdx = i;
                    RunPhase(() => BuildArchitecturalWall(buildingData.walls[wallIdx]));
                    if (i % 4 == 0)
                    {
                        ShowLoadingHUD($"Constructing walls… ({i+1}/{total})", 0.40f + 0.30f * i / total);
                        yield return null;
                    }
                }
            }

            // ── Phase 6: Door frames ──────────────────────────
            ShowLoadingHUD("Placing door frames & openings…", 0.72f);
            yield return null;

            if (buildingData?.doors != null && phaseException == null)
            {
                RunPhase(() =>
                {
                    foreach (var d in buildingData.doors) BuildDoorFrame(d);
                });
            }
            yield return null;

            // ── Phase 7: Window frames ────────────────────────
            ShowLoadingHUD("Adding window frames & glazing…", 0.80f);
            yield return null;

            if (buildingData?.windows != null && phaseException == null)
            {
                RunPhase(() =>
                {
                    foreach (var w in buildingData.windows) BuildWindowFrame(w);
                });
            }
            yield return null;

            // ── Phase 8: Ceiling ──────────────────────────────
            ShowLoadingHUD("Generating ceiling geometry…", 0.87f);
            yield return null;

            RunPhase(() => BuildCeiling(bCenter, bW, bD));
            yield return null;

            // ── Phase 9: Room lights ──────────────────────────
            ShowLoadingHUD("Baking interior lighting…", 0.92f);
            yield return null;

            RunPhase(() => BuildRoomLights());
            yield return null;

            // ── Phase 10: Trees / landscape ───────────────────
            ShowLoadingHUD("Populating landscape assets…", 0.96f);
            yield return null;

            RunPhase(() => BuildEnvironmentTrees(bCenter, bW, bD));
            yield return null;

            // ── Phase 11: Position player ─────────────────────
            ShowLoadingHUD("Positioning player controller…", 0.99f);
            yield return null;

            RunPhase(() => PositionPlayer(bCenter, bW, bD));

            if (phaseException != null)
            {
                Debug.LogError($"[BuildingLoader] Exception during async loading: {phaseException.Message}\n{phaseException.StackTrace}");
                ShowLoadingHUD("Generating recovery model…", 0.99f);
                yield return null;
                try
                {
                    buildingData = GenerateDefaultBuildingJson();
                    ClearBuilding();
                    BuildMaterials();
                    GetBuildingBounds(out bCenter, out bW, out bD);
                    BuildOutdoorEnvironment(bCenter, bW, bD);
                    BuildFloors(bCenter, bW, bD);
                    if (buildingData.walls != null)
                        foreach (var w in buildingData.walls) BuildArchitecturalWall(w);
                    BuildCeiling(bCenter, bW, bD);
                    PositionPlayer(bCenter, bW, bD);
                }
                catch (Exception ex2)
                {
                    Debug.LogError($"[BuildingLoader] Fallback build failed: {ex2.Message}");
                }
            }

            ShowLoadingHUD("", 1f);
            HideLoadingHUD();
            isLoading = false;
            Debug.Log("[BuildingLoader] ✓ 3D architectural model load cycle completed.");
        }

        // ══════════════════════════════════════════════════════
        //  MATERIAL FACTORY
        // ══════════════════════════════════════════════════════
        private void BuildMaterials()
        {
            matOuterWall    = MakeMat(new Color(0.80f, 0.77f, 0.72f), 0.08f);   // warm stucco
            matInnerWall    = MakeMat(new Color(0.91f, 0.90f, 0.88f), 0.04f);   // clean plaster
            matFloor        = MakeMat(new Color(0.62f, 0.52f, 0.40f), 0.25f);   // polished hardwood
            matCeiling      = MakeMat(new Color(0.94f, 0.94f, 0.95f), 0.02f);   // matte white
            matDoor         = MakeMat(new Color(0.40f, 0.25f, 0.13f), 0.30f);   // natural timber
            matWindowFrame  = MakeMat(new Color(0.92f, 0.92f, 0.92f), 0.20f);   // white PVC frame
            matWindowGlass  = MakeGlassMat();                                     // transparent glazing
            matGrass        = MakeMat(new Color(0.22f, 0.44f, 0.18f), 0.03f);
            matPavement     = MakeMat(new Color(0.38f, 0.37f, 0.36f), 0.15f);
            matWalkway      = MakeMat(new Color(0.65f, 0.60f, 0.54f), 0.12f);
            matTrunkBark    = MakeMat(new Color(0.33f, 0.21f, 0.09f), 0.04f);
            matLeaf         = MakeMat(new Color(0.16f, 0.36f, 0.12f), 0.04f);
        }

        private Material MakeGlassMat()
        {
            Color glassColor = new Color(0.65f, 0.85f, 0.96f, 0.18f);   // 18% alpha clear sky tint
            Material template = Resources.Load<Material>("GlassMaterial");
            Material mat;
            if (template != null)
            {
                mat = new Material(template);
            }
            else
            {
                Shader s = Shader.Find("Standard");
                if (s == null) s = Shader.Find("Unlit/Transparent");
                if (s == null) s = Shader.Find("Universal Render Pipeline/Unlit");
                if (s == null) s = Shader.Find("Sprites/Default");
                mat = new Material(s);
            }

            // Configure Built-in Standard shader for transparency
            mat.SetFloat("_Mode", 3.0f);   // 3 = Transparent
            mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
            mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            mat.SetInt("_ZWrite", 0);
            mat.DisableKeyword("_ALPHATEST_ON");
            mat.DisableKeyword("_ALPHABLEND_ON");
            mat.EnableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent + 100;
            mat.SetOverrideTag("RenderType", "Transparent");

            if (mat.HasProperty("_Color"))     mat.SetColor("_Color",     glassColor);
            if (mat.HasProperty("_BaseColor")) mat.SetColor("_BaseColor", glassColor);
            mat.SetFloat("_Surface", 1.0f);
            mat.SetFloat("_Blend", 0.0f);
            mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            mat.color = glassColor;
            return mat;
        }

        private float GetWindowWidth(WindowJson wn, float wallLen)
        {
            float winSpanPx = Mathf.Max(wn.width_px, wn.height_px);
            float baseSpan = (winSpanPx > 0f) ? (winSpanPx / pxPerM) * modelScaleMultiplier : 1.40f;
            float winW = Mathf.Clamp(baseSpan, 1.10f, 2.20f);
            if (wallLen > 0.80f && winW > wallLen - 0.50f)
            {
                winW = Mathf.Max(0.70f, wallLen - 0.50f);
            }
            return winW;
        }

        private WallJson FindHostWall(float wx, float wz, float expectedAngleDeg = -1f, bool isWindow = false)
        {
            if (buildingData?.walls == null || buildingData.walls.Length == 0) return null;

            WallJson bestWall = null;
            float minScore = float.MaxValue;
            Vector3 p = new Vector3(wx, 0f, wz);

            Vector3 expectedDir = Vector3.zero;
            bool checkAngle = (expectedAngleDeg >= 0f);
            if (checkAngle)
            {
                float rad = expectedAngleDeg * Mathf.Deg2Rad;
                expectedDir = new Vector3(Mathf.Cos(rad), 0f, -Mathf.Sin(rad)).normalized;
            }

            foreach (var w in buildingData.walls)
            {
                float x1 = (w.start[0] / pxPerM) * modelScaleMultiplier;
                float z1 = ((imgHeight - w.start[1]) / pxPerM) * modelScaleMultiplier;
                float x2 = (w.end[0] / pxPerM) * modelScaleMultiplier;
                float z2 = ((imgHeight - w.end[1]) / pxPerM) * modelScaleMultiplier;

                Vector3 a = new Vector3(x1, 0f, z1);
                Vector3 b = new Vector3(x2, 0f, z2);

                Vector3 ab = b - a;
                float len = ab.magnitude;
                if (len < 0.25f) continue; // Skip tiny degraded fragments

                Vector3 dir = ab / len;

                // Collinearity check: reject walls running perpendicular to the feature
                if (checkAngle)
                {
                    float collinearity = Mathf.Abs(Vector3.Dot(dir, expectedDir));
                    if (collinearity < 0.70f) continue;
                }

                float rawT = Vector3.Dot(p - a, dir); // metres along wall
                Vector3 projSeg = a + dir * Mathf.Clamp(rawT, 0f, len);
                float distSeg = Vector3.Distance(p, projSeg);

                // Penalty if feature projection falls outside the host wall segment
                float outPenalty = (rawT < -0.20f || rawT > len + 0.20f) ? 50f : 0f;
                // Prefer outer walls for windows
                float outerBonus = (isWindow && w.wall_type == "outer") ? -0.5f : 0f;
                // Prefer longer continuous walls over fragmented slivers
                float lenBonus = -0.05f * Mathf.Min(len, 10f);

                float score = distSeg + outPenalty + outerBonus + lenBonus;

                if (score < minScore)
                {
                    minScore = score;
                    bestWall = w;
                }
            }

            return bestWall;
        }

        private Material MakeMat(Color color, float smoothness = 0.1f)
        {
            Shader s = FindShader("Universal Render Pipeline/Lit", "Universal Render Pipeline/Simple Lit", "Standard", "Unlit/Color", "Sprites/Default");

            var mat = new Material(s);
            if (mat.HasProperty("_BaseColor"))  mat.SetColor("_BaseColor", color);
            if (mat.HasProperty("_Color"))      mat.SetColor("_Color",     color);
            if (mat.HasProperty("_Smoothness")) mat.SetFloat("_Smoothness", smoothness);
            if (mat.HasProperty("_Glossiness")) mat.SetFloat("_Glossiness", smoothness);
            mat.color = color;

            if (color.a < 0.99f)
            {
                mat.SetFloat("_Mode", 3.0f);   // Transparent
                mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.SetInt("_ZWrite", 0);
                mat.DisableKeyword("_ALPHATEST_ON");
                mat.DisableKeyword("_ALPHABLEND_ON");
                mat.EnableKeyword("_ALPHAPREMULTIPLY_ON");
                mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                mat.SetOverrideTag("RenderType", "Transparent");

                mat.SetFloat("_Surface", 1.0f);   // Transparent
                mat.SetFloat("_Blend", 0.0f);     // Alpha blend
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            }
            return mat;
        }

        // ══════════════════════════════════════════════════════
        //  SCENE LIGHTING
        // ══════════════════════════════════════════════════════
#pragma warning disable 0618
        private void ConfigureSceneLighting()
        {
#if UNITY_2023_1_OR_NEWER
            foreach (var l in FindObjectsByType<Light>(FindObjectsSortMode.None))
#else
            foreach (var l in FindObjectsOfType<Light>())
#endif
            {
                if (l.type == LightType.Directional)
                {
                    l.intensity     = 1.1f;
                    l.color         = new Color(1.0f, 0.95f, 0.85f);
                    l.shadows       = LightShadows.Soft;
                    l.shadowStrength = 0.60f;
                    l.transform.rotation = Quaternion.Euler(42f, -35f, 0f);
                }
            }

            RenderSettings.ambientMode  = UnityEngine.Rendering.AmbientMode.Trilight;
            RenderSettings.ambientSkyColor     = new Color(0.52f, 0.70f, 0.96f);
            RenderSettings.ambientEquatorColor = new Color(0.48f, 0.52f, 0.56f);
            RenderSettings.ambientGroundColor  = new Color(0.22f, 0.20f, 0.16f);
            RenderSettings.ambientIntensity    = 1.05f;
            RenderSettings.fogColor = new Color(0.68f, 0.82f, 0.98f);
            RenderSettings.fogMode  = FogMode.ExponentialSquared;
            RenderSettings.fogDensity = 0.003f;
            RenderSettings.fog = true;

            AddSceneLight(LightType.Directional,
                new Vector3(0, 0, 0), new Color(0.55f, 0.68f, 0.90f), 0.25f,
                Quaternion.Euler(-90f, 0f, 0f));
        }

        private void AddSceneLight(LightType type, Vector3 pos, Color color, float intensity, Quaternion rot)
        {
            var go = new GameObject("SceneFillLight");
            go.transform.position = pos;
            go.transform.rotation = rot;
            var l = go.AddComponent<Light>();
            l.type      = type;
            l.color     = color;
            l.intensity = intensity;
            l.shadows   = LightShadows.None;
        }

        private void ConfigureCamera()
        {
            Camera cam = Camera.main;
#if UNITY_2023_1_OR_NEWER
            if (cam == null) cam = FindFirstObjectByType<Camera>();
#else
            if (cam == null) cam = FindObjectOfType<Camera>();
#endif
            if (cam == null) return;

            cam.allowHDR    = true;
            cam.farClipPlane = 1200f;
            cam.nearClipPlane = 0.08f;
            cam.clearFlags  = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.40f, 0.62f, 0.92f);
        }
#pragma warning restore 0618

        private void BuildOutdoorEnvironment(Vector3 bCenter, float bW, float bD)
        {
            // Grass terrain
            SpawnCube("Terrain_Grass", generatedBuildingRoot,
                new Vector3(bCenter.x, -0.31f, bCenter.z),
                new Vector3(Mathf.Max(600f, bW * 8f), 0.50f, Mathf.Max(600f, bD * 8f)),
                matGrass);

            // Patio slab
            SpawnCube("Patio_Slab", generatedBuildingRoot,
                new Vector3(bCenter.x, -0.07f, bCenter.z),
                new Vector3(bW + 8f, 0.10f, bD + 8f),
                matPavement);

            // Walkway
            float southZ = bCenter.z - bD * 0.5f;
            SpawnCube("Front_Walkway", generatedBuildingRoot,
                new Vector3(bCenter.x, -0.03f, southZ - 4.0f),
                new Vector3(3.5f, 0.04f, 8f),
                matWalkway);
        }

        // ══════════════════════════════════════════════════════
        //  FLOOR SLABS
        // ══════════════════════════════════════════════════════
        private void BuildFloors(Vector3 bCenter, float bW, float bD)
        {
            SpawnCube("BuildingFloor_Main", generatedBuildingRoot,
                new Vector3(bCenter.x, 0.01f, bCenter.z),
                new Vector3(bW + 0.6f, 0.04f, bD + 0.6f),
                matFloor);
        }

        private float GetSnappedDoorT(DoorJson d, WallJson hostWall, float length, Vector3 startV, Vector3 dir, float doorW)
        {
            float dx = (d.center[0] / pxPerM) * modelScaleMultiplier;
            float dz = ((imgHeight - d.center[1]) / pxPerM) * modelScaleMultiplier;
            Vector3 dPos = new Vector3(dx, 0f, dz);

            float t = Vector3.Dot(dPos - startV, dir) / length;
            t = Mathf.Clamp(t, 0.15f, 0.85f);

            if (buildingData.windows != null)
            {
                foreach (var wn in buildingData.windows)
                {
                    float wx = (wn.center[0] / pxPerM) * modelScaleMultiplier;
                    float wz = ((imgHeight - wn.center[1]) / pxPerM) * modelScaleMultiplier;
                    Vector3 wPos = new Vector3(wx, 0f, wz);

                    float wt = Vector3.Dot(wPos - startV, dir) / length;
                    Vector3 wproj = startV + dir * Mathf.Clamp01(wt) * length;
                    if (Vector3.Distance(wPos, wproj) < 0.6f && wt >= 0f && wt <= 1f)
                    {
                        float winSpanPx = (wn.width_px > 0f) ? wn.width_px : 80f;
                        float winW = Mathf.Clamp(winSpanPx / pxPerM, 0.85f, 1.20f);
                        float minDistT = (doorW * 0.5f + winW * 0.5f + 0.40f) / length;
                        if (Mathf.Abs(t - wt) < minDistT)
                        {
                            if (t >= wt) t = Mathf.Min(0.85f, wt + minDistT);
                            else         t = Mathf.Max(0.15f, wt - minDistT);
                        }
                    }
                }
            }

            return t;
        }

        // ══════════════════════════════════════════════════════
        //  WALL BUILDER — seamless thick walls with 0 corner gaps
        // ══════════════════════════════════════════════════════
        private void BuildArchitecturalWall(WallJson w)
        {
            bool isOuter = (w.wall_type == "outer");

            float x1 = (w.start[0] / pxPerM) * modelScaleMultiplier;
            float z1 = ((imgHeight - w.start[1]) / pxPerM) * modelScaleMultiplier;
            float x2 = (w.end[0]   / pxPerM) * modelScaleMultiplier;
            float z2 = ((imgHeight - w.end[1])   / pxPerM) * modelScaleMultiplier;

            Vector3 startV = new Vector3(x1, 0f, z1);
            Vector3 endV   = new Vector3(x2, 0f, z2);
            float length   = Vector3.Distance(startV, endV);
            if (length < 0.10f) return;

            float rawThickness = (w.thickness_px / pxPerM) * modelScaleMultiplier;
            float minT = isOuter ? minOuterWallThickness : minInternalWallThickness;
            float thickness = Mathf.Max(rawThickness, minT);

            Vector3 dir = (endV - startV).normalized;
            Quaternion rot = Quaternion.LookRotation(dir, Vector3.up) * Quaternion.Euler(0f, 90f, 0f);

            // Collect door gaps on this wall segment
            List<Vector2> doorGaps = new List<Vector2>();
            if (buildingData.doors != null)
            {
                foreach (var d in buildingData.doors)
                {
                    // Check if door orientation matches this wall direction
                    float rad = d.angle_deg * Mathf.Deg2Rad;
                    Vector3 dDir = new Vector3(Mathf.Cos(rad), 0f, -Mathf.Sin(rad)).normalized;
                    float collinearity = Mathf.Abs(Vector3.Dot(dir, dDir));
                    if (collinearity < 0.70f) continue; // Skip perpendicular adjoining walls

                    float dx = (d.center[0] / pxPerM) * modelScaleMultiplier;
                    float dz = ((imgHeight - d.center[1]) / pxPerM) * modelScaleMultiplier;
                    Vector3 dPos = new Vector3(dx, 0f, dz);

                    float rawT = Vector3.Dot(dPos - startV, dir) / length;
                    Vector3 proj = startV + dir * Mathf.Clamp01(rawT) * length;
                    float distToLine = Vector3.Distance(dPos, proj);

                    float doorTol = thickness * 0.5f + 0.35f;
                    if (distToLine < doorTol && rawT >= 0.01f && rawT <= 0.99f)
                    {
                        float doorW = 1.00f;               // Standard clear door passage width 1.00m
                        float frameT = 0.06f;              // Frame jamb thickness 6cm
                        float clearance = 0.02f;           // 2cm clearance on each side
                        float gapW = doorW + frameT * 2f + clearance * 2f; // 1.16m total cutout width in wall
                        float t = GetSnappedDoorT(d, w, length, startV, dir, doorW);
                        float t_half = (gapW * 0.5f) / length;

                        float ta = Mathf.Clamp(t - t_half, 0.0f, 1.0f);
                        float tb = Mathf.Clamp(t + t_half, 0.0f, 1.0f);
                        if (tb - ta > 0.01f)
                        {
                            doorGaps.Add(new Vector2(ta, tb));
                        }
                    }
                }
            }

            // Collect window openings on this wall segment
            List<Vector2> winGaps = new List<Vector2>();
            if (buildingData.windows != null)
            {
                foreach (var wn in buildingData.windows)
                {
                    // Check if window orientation matches this wall direction
                    float rad = wn.angle_deg * Mathf.Deg2Rad;
                    Vector3 wDir = new Vector3(Mathf.Cos(rad), 0f, -Mathf.Sin(rad)).normalized;
                    float collinearity = Mathf.Abs(Vector3.Dot(dir, wDir));
                    if (collinearity < 0.70f) continue; // Skip perpendicular adjoining walls

                    float wx = (wn.center[0] / pxPerM) * modelScaleMultiplier;
                    float wz = ((imgHeight - wn.center[1]) / pxPerM) * modelScaleMultiplier;
                    Vector3 wPos = new Vector3(wx, 0f, wz);

                    float t = Vector3.Dot(wPos - startV, dir) / length;
                    if (t >= 0.01f && t <= 0.99f)
                    {
                        Vector3 proj = startV + dir * (t * length);
                        if (Vector3.Distance(wPos, proj) < thickness * 0.5f + 0.35f)
                        {
                            float winW = GetWindowWidth(wn, length);
                            float frameT = 0.06f;
                            float clearance = 0.02f;       // 2cm clearance on each side
                            float gapW = winW + frameT * 2f + clearance * 2f;
                            float t_half = (gapW * 0.5f) / length;

                            float ta = Mathf.Clamp(t - t_half, 0.01f, 0.99f);
                            float tb = Mathf.Clamp(t + t_half, 0.01f, 0.99f);
                            if (tb - ta > 0.02f)
                            {
                                winGaps.Add(new Vector2(ta, tb));
                            }
                        }
                    }
                }
            }

            // Helper to spawn a wall sub-box with corner endpoint extensions
            void SpawnSubWall(string name, float tStart, float tEnd, float yMin, float yMax)
            {
                // Extend outer corner endpoints by 0.75x thickness so corner joints overlap seamlessly (zero corner gaps!)
                float extStart = (tStart <= 0.001f) ? -(thickness * 0.75f) / length : tStart;
                float extEnd   = (tEnd >= 0.999f)   ? 1.0f + (thickness * 0.75f) / length : tEnd;

                float subLen = (extEnd - extStart) * length;
                if (subLen < 0.05f || yMax - yMin < 0.05f) return;

                Vector3 subStart = startV + dir * (extStart * length);
                Vector3 subEnd   = startV + dir * (extEnd * length);
                Vector3 subCenter = (subStart + subEnd) * 0.5f + new Vector3(0f, (yMin + yMax) * 0.5f, 0f);
                float subH = yMax - yMin;

                Mesh subMesh = BuildBoxMesh(subLen, subH, thickness);
                var subGO = new GameObject(name);
                subGO.transform.parent   = generatedBuildingRoot;
                subGO.transform.position = subCenter;
                subGO.transform.rotation = rot;

                var mf = subGO.AddComponent<MeshFilter>();
                mf.sharedMesh = subMesh;
                var mr = subGO.AddComponent<MeshRenderer>();
                mr.sharedMaterial = isOuter ? matOuterWall : matInnerWall;
                mr.receiveShadows = true;
                mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.TwoSided;
                var mc = subGO.AddComponent<MeshCollider>();
                mc.sharedMesh = subMesh;
            }

            float doorH = 2.10f;
            float doorCutoutH = doorH + 0.06f + 0.02f; // Door lintel cutout with 2cm clearance (2.18m)
            float sillH = 0.90f;                       // Standard window sill height (0.90m)
            float winH  = 1.30f;                       // Standard window height (1.30m)
            float headH = sillH + winH + 0.06f * 2f + 0.04f; // Window head opening with 2cm clearance (2.36m)

            // Subdivide wall length [0, 1] around door and window intervals
            List<float> splitPoints = new List<float> { 0f, 1f };
            foreach (var g in doorGaps) { splitPoints.Add(g.x); splitPoints.Add(g.y); }
            foreach (var g in winGaps)  { splitPoints.Add(g.x); splitPoints.Add(g.y); }
            splitPoints.Sort();

            for (int i = 0; i < splitPoints.Count - 1; i++)
            {
                float ta = splitPoints[i];
                float tb = splitPoints[i + 1];
                if (tb - ta < 1e-4f) continue;

                float tMid = (ta + tb) * 0.5f;
                bool isDoor = false;
                foreach (var g in doorGaps)
                    if (tMid >= g.x && tMid <= g.y) { isDoor = true; break; }

                bool isWin = false;
                foreach (var g in winGaps)
                    if (tMid >= g.x && tMid <= g.y) { isWin = true; break; }

                if (isDoor)
                {
                    // Lintel wall piece above door:
                    SpawnSubWall($"Wall_{w.id}_Lintel", ta, tb, doorCutoutH, wallHeight);
                }
                else if (isWin)
                {
                    // Window opening: sill piece + head piece
                    SpawnSubWall($"Wall_{w.id}_Sill", ta, tb, 0f, sillH);
                    SpawnSubWall($"Wall_{w.id}_Head", ta, tb, headH, wallHeight);
                }
                else
                {
                    // Solid full-height wall piece
                    SpawnSubWall($"Wall_{w.id}_Solid", ta, tb, 0f, wallHeight);
                }
            }
        }

        // ══════════════════════════════════════════════════════
        //  DOOR FRAMES
        // ══════════════════════════════════════════════════════
        private void BuildDoorFrame(DoorJson d)
        {
            float wx = (d.center[0] / pxPerM) * modelScaleMultiplier;
            float wz = ((imgHeight - d.center[1]) / pxPerM) * modelScaleMultiplier;

            WallJson hostWall = FindHostWall(wx, wz, d.angle_deg, isWindow: false);
            bool isOuterDoor = (hostWall != null && hostWall.wall_type == "outer");

            float doorW = 1.00f; // Clear passage width (1.00m)
            Vector3 dir;
            float wallThickness = isOuterDoor ? minOuterWallThickness : minInternalWallThickness;

            if (hostWall != null)
            {
                float x1 = (hostWall.start[0] / pxPerM) * modelScaleMultiplier;
                float z1 = ((imgHeight - hostWall.start[1]) / pxPerM) * modelScaleMultiplier;
                float x2 = (hostWall.end[0] / pxPerM) * modelScaleMultiplier;
                float z2 = ((imgHeight - hostWall.end[1]) / pxPerM) * modelScaleMultiplier;
                Vector3 w1 = new Vector3(x1, 0f, z1);
                Vector3 w2 = new Vector3(x2, 0f, z2);
                dir = (w2 - w1).normalized;
                float hLen = Vector3.Distance(w1, w2);
                if (hLen > 1e-4f)
                {
                    float projT = GetSnappedDoorT(d, hostWall, hLen, w1, dir, doorW);
                    Vector3 snappedPos = w1 + dir * Mathf.Clamp01(projT) * hLen;
                    wx = snappedPos.x;
                    wz = snappedPos.z;
                }
                float rawThickness = (hostWall.thickness_px / pxPerM) * modelScaleMultiplier;
                float minT = isOuterDoor ? minOuterWallThickness : minInternalWallThickness;
                wallThickness = Mathf.Max(rawThickness, minT);
            }
            else
            {
                float rad = d.angle_deg * Mathf.Deg2Rad;
                dir = new Vector3(Mathf.Cos(rad), 0f, -Mathf.Sin(rad)).normalized;
            }

            float doorH = 2.10f; // Standard door height (2.10m)
            float frameT = 0.06f; // Frame jamb thickness (6cm)
            float frameDepth = wallThickness + 0.02f; // Wall thickness + 1cm casing overlap on each side

            Quaternion rot = Quaternion.LookRotation(dir, Vector3.up) * Quaternion.Euler(0f, 90f, 0f);

            string doorTag = isOuterDoor ? "OuterDoor" : "InnerDoor";
            var frameRoot = new GameObject($"{doorTag}_{d.id}");
            frameRoot.transform.parent   = generatedBuildingRoot;
            frameRoot.transform.position = new Vector3(wx, 0f, wz);
            frameRoot.transform.rotation = rot;

            // Left jamb
            SpawnCube("Jamb_L", frameRoot.transform,
                new Vector3(-(doorW * 0.5f + frameT * 0.5f), doorH * 0.5f, 0f),
                new Vector3(frameT, doorH, frameDepth),
                matDoor);

            // Right jamb
            SpawnCube("Jamb_R", frameRoot.transform,
                new Vector3(doorW * 0.5f + frameT * 0.5f, doorH * 0.5f, 0f),
                new Vector3(frameT, doorH, frameDepth),
                matDoor);

            // Lintel
            SpawnCube("Lintel", frameRoot.transform,
                new Vector3(0f, doorH + frameT * 0.5f, 0f),
                new Vector3(doorW + frameT * 2f, frameT, frameDepth),
                matDoor);

            // Hinge pivot at left jamb edge
            var pivot = new GameObject("Pivot");
            pivot.transform.SetParent(frameRoot.transform, false);
            pivot.transform.localPosition = new Vector3(-doorW * 0.5f, 0f, 0f);
            pivot.transform.localRotation = Quaternion.identity;

            var dc = pivot.AddComponent<DoorController>();
            dc.doorId = d.id;
            dc.isOpen = false;
            dc.openAngle = 90f;
            dc.openSpeed = 5f;

            // Door panel — closed by default inside the door frame
            var panel = SpawnCube("Door_Panel", pivot.transform,
                new Vector3(doorW * 0.5f, doorH * 0.5f, 0f),
                new Vector3(doorW - 0.01f, doorH - 0.02f, 0.045f),
                matDoor);

            if (panel.GetComponent<Collider>() == null)
                panel.AddComponent<BoxCollider>();

            Material matHandle = MakeMat(new Color(0.85f, 0.70f, 0.35f), 0.90f);
            SpawnCube("Handle_Front", panel.transform,
                new Vector3(doorW * 0.40f, 0f, 0.035f),
                new Vector3(0.12f, 0.03f, 0.04f),
                matHandle);
            SpawnCube("Handle_Back", panel.transform,
                new Vector3(doorW * 0.40f, 0f, -0.035f),
                new Vector3(0.12f, 0.03f, 0.04f),
                matHandle);
        }

        // ══════════════════════════════════════════════════════
        //  WINDOW FRAMES  — snapped flat along host wall direction
        // ══════════════════════════════════════════════════════
        private void BuildWindowFrame(WindowJson wn)
        {
            float wx = (wn.center[0] / pxPerM) * modelScaleMultiplier;
            float wz = ((imgHeight - wn.center[1]) / pxPerM) * modelScaleMultiplier;

            WallJson hostWall = FindHostWall(wx, wz, wn.angle_deg, isWindow: true);
            Vector3 dir;
            float wallThickness = minOuterWallThickness;
            float hLen = 10f;

            if (hostWall != null)
            {
                float x1 = (hostWall.start[0] / pxPerM) * modelScaleMultiplier;
                float z1 = ((imgHeight - hostWall.start[1]) / pxPerM) * modelScaleMultiplier;
                float x2 = (hostWall.end[0] / pxPerM) * modelScaleMultiplier;
                float z2 = ((imgHeight - hostWall.end[1]) / pxPerM) * modelScaleMultiplier;
                Vector3 w1 = new Vector3(x1, 0f, z1);
                Vector3 w2 = new Vector3(x2, 0f, z2);
                dir = (w2 - w1).normalized;
                hLen = Vector3.Distance(w1, w2);
                if (hLen > 1e-4f)
                {
                    Vector3 wPos = new Vector3(wx, 0f, wz);
                    float projT = Vector3.Dot(wPos - w1, dir) / hLen;
                    projT = Mathf.Clamp(projT, 0.15f, 0.85f);
                    Vector3 snappedPos = w1 + dir * (projT * hLen);
                    wx = snappedPos.x;
                    wz = snappedPos.z;
                }
                float rawThickness = (hostWall.thickness_px / pxPerM) * modelScaleMultiplier;
                wallThickness = Mathf.Max(rawThickness, minOuterWallThickness);
            }
            else
            {
                float rad = wn.angle_deg * Mathf.Deg2Rad;
                dir = new Vector3(Mathf.Cos(rad), 0f, -Mathf.Sin(rad)).normalized;
            }

            float winW = GetWindowWidth(wn, hLen);
            float winH = 1.30f;                  // Standard 3D architectural window vertical height
            float sillH = 0.90f;                 // Standard 90cm sill height
            float frameT = 0.06f;

            float frameDepth = wallThickness + 0.02f; // Fits host wall thickness cleanly
            Quaternion rot = Quaternion.LookRotation(dir, Vector3.up) * Quaternion.Euler(0f, 90f, 0f);

            var fRoot = new GameObject($"Window_{wn.id}");
            fRoot.transform.parent   = generatedBuildingRoot;
            fRoot.transform.position = new Vector3(wx, 0f, wz);
            fRoot.transform.rotation = rot;

            float centerY = sillH + winH * 0.5f;

            // Left jamb
            SpawnCube("WJamb_L", fRoot.transform,
                new Vector3(-(winW * 0.5f + frameT * 0.5f), centerY, 0f),
                new Vector3(frameT, winH + frameT, frameDepth),
                matWindowFrame);

            // Right jamb
            SpawnCube("WJamb_R", fRoot.transform,
                new Vector3(winW * 0.5f + frameT * 0.5f, centerY, 0f),
                new Vector3(frameT, winH + frameT, frameDepth),
                matWindowFrame);

            // Top head
            SpawnCube("WHead", fRoot.transform,
                new Vector3(0f, sillH + winH + frameT * 0.5f, 0f),
                new Vector3(winW + frameT * 2f, frameT, frameDepth),
                matWindowFrame);

            // Sill
            SpawnCube("WSill", fRoot.transform,
                new Vector3(0f, sillH - frameT * 0.5f, 0f),
                new Vector3(winW + frameT * 2f, frameT, frameDepth * 1.1f),
                matWindowFrame);

            // Mid rail
            SpawnCube("WRail", fRoot.transform,
                new Vector3(0f, sillH + winH * 0.5f, 0f),
                new Vector3(winW, frameT * 0.8f, frameDepth * 0.8f),
                matWindowFrame);

            // Glass pane
            var gObj = SpawnCube("WGlass", fRoot.transform,
                new Vector3(0f, centerY, 0f),
                new Vector3(winW, winH, 0.03f),
                matWindowGlass);
            var mr = gObj.GetComponent<MeshRenderer>();
            if (mr != null) mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }

        // ══════════════════════════════════════════════════════
        //  CEILING SLAB
        // ══════════════════════════════════════════════════════
        private void BuildCeiling(Vector3 bCenter, float bW, float bD)
        {
            ceilingContainer = new GameObject("CeilingGroup");
            ceilingContainer.transform.parent = generatedBuildingRoot;

            var cObj = SpawnCube("Ceiling_Main", ceilingContainer.transform,
                new Vector3(bCenter.x, wallHeight + 0.08f, bCenter.z),
                new Vector3(bW + 0.6f, 0.16f, bD + 0.6f),
                matCeiling);
            cObj.SetActive(showCeiling);

            var roofMat = MakeMat(new Color(0.50f, 0.40f, 0.32f), 0.10f);
            var rObj = SpawnCube("Roof_Overhang", ceilingContainer.transform,
                new Vector3(bCenter.x, wallHeight + 0.32f, bCenter.z),
                new Vector3(bW + 2.0f, 0.24f, bD + 2.0f),
                roofMat);
            rObj.SetActive(showCeiling);
        }

        // ══════════════════════════════════════════════════════
        //  ROOM POINT LIGHTS
        // ══════════════════════════════════════════════════════
        private void BuildRoomLights()
        {
            if (buildingData.rooms == null) return;

            foreach (var r in buildingData.rooms)
            {
                float rx = (r.centroid[0] / pxPerM) * modelScaleMultiplier;
                float rz = ((imgHeight - r.centroid[1]) / pxPerM) * modelScaleMultiplier;

                var lGO = new GameObject($"RoomLight_{r.label}");
                lGO.transform.parent   = generatedBuildingRoot;
                lGO.transform.position = new Vector3(rx, wallHeight - 0.35f, rz);

                var pl = lGO.AddComponent<Light>();
                pl.type      = LightType.Point;
                pl.color     = new Color(1.0f, 0.92f, 0.82f);
                pl.intensity = 1.20f;
                pl.range     = Mathf.Max(10f, Mathf.Sqrt(r.area_m2 > 0 ? r.area_m2 : 20f) * modelScaleMultiplier * 1.6f);
                pl.shadows   = LightShadows.Soft;
                pl.shadowStrength = 0.40f;

                var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                sphere.name = "LampBulb";
                sphere.transform.parent   = lGO.transform;
                sphere.transform.localPosition = Vector3.zero;
                sphere.transform.localScale    = new Vector3(0.15f, 0.15f, 0.15f);
                Destroy(sphere.GetComponent<Collider>());
                var lampMat = MakeMat(new Color(1.0f, 0.96f, 0.82f), 0.95f);
                sphere.GetComponent<Renderer>().sharedMaterial = lampMat;
            }
        }

        // ══════════════════════════════════════════════════════
        //  ENVIRONMENT TREES
        // ══════════════════════════════════════════════════════
        private void BuildEnvironmentTrees(Vector3 bCenter, float bW, float bD)
        {
            var treeRoot = new GameObject("EnvironmentTrees");
            treeRoot.transform.parent = generatedBuildingRoot;

            Vector3 center  = new Vector3(bCenter.x, 0f, bCenter.z);
            float   radius  = Mathf.Max(bW, bD) * 0.8f + 15f;
            int     count   = 24;

            System.Random rng = new System.Random(42);

            for (int i = 0; i < count; i++)
            {
                float angle = i * (360f / count) * Mathf.Deg2Rad;
                float r     = radius + (float)(rng.NextDouble() * 6.0);
                float tx    = center.x + Mathf.Cos(angle) * r;
                float tz    = center.z + Mathf.Sin(angle) * r;

                float scale  = 1.0f + (float)(rng.NextDouble() * 0.5);
                float trunkH = 3.5f * scale;
                float leafS  = 5.0f * scale;

                var trunk = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                trunk.name = $"Trunk_{i}";
                trunk.transform.parent   = treeRoot.transform;
                trunk.transform.position = new Vector3(tx, trunkH * 0.5f, tz);
                trunk.transform.localScale = new Vector3(0.5f * scale, trunkH * 0.5f, 0.5f * scale);
                trunk.GetComponent<Renderer>().sharedMaterial = matTrunkBark;

                var canopy = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                canopy.name = $"Canopy_{i}";
                canopy.transform.parent   = treeRoot.transform;
                canopy.transform.position = new Vector3(tx, trunkH + leafS * 0.4f, tz);
                canopy.transform.localScale = new Vector3(leafS, leafS * 1.15f, leafS);
                canopy.GetComponent<Renderer>().sharedMaterial = matLeaf;
                Destroy(canopy.GetComponent<Collider>());
            }

            for (int i = 0; i < 4; i++)
            {
                float lx = bW * 0.5f + (i % 2 == 0 ? 2.2f : -2.2f);
                float lz = -2.0f - i * 2.5f;
                BuildGardenLamp(treeRoot.transform, new Vector3(lx, 0f, lz));
            }
        }

        private void BuildGardenLamp(Transform parent, Vector3 pos)
        {
            var lampRoot = new GameObject("GardenLamp");
            lampRoot.transform.parent   = parent;
            lampRoot.transform.position = pos;

            var post = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            post.transform.parent   = lampRoot.transform;
            post.transform.localPosition = new Vector3(0f, 1.25f, 0f);
            post.transform.localScale    = new Vector3(0.08f, 1.25f, 0.08f);
            post.GetComponent<Renderer>().sharedMaterial = MakeMat(new Color(0.15f, 0.15f, 0.16f), 0.60f);
            Destroy(post.GetComponent<Collider>());

            var globe = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            globe.transform.parent   = lampRoot.transform;
            globe.transform.localPosition = new Vector3(0f, 2.6f, 0f);
            globe.transform.localScale    = new Vector3(0.35f, 0.35f, 0.35f);
            globe.GetComponent<Renderer>().sharedMaterial = MakeMat(new Color(1f, 0.95f, 0.80f, 0.8f), 0.85f);
            Destroy(globe.GetComponent<Collider>());

            var lGO = new GameObject("LampLight");
            lGO.transform.parent   = lampRoot.transform;
            lGO.transform.localPosition = new Vector3(0f, 2.6f, 0f);
            var pl = lGO.AddComponent<Light>();
            pl.type      = LightType.Point;
            pl.color     = new Color(1.0f, 0.88f, 0.60f);
            pl.intensity = 0.80f;
            pl.range     = 8f;
            pl.shadows   = LightShadows.None;
        }

        // ══════════════════════════════════════════════════════
        //  PLAYER POSITIONING
        // ══════════════════════════════════════════════════════
        private void PositionPlayer(Vector3 bCenter, float bW, float bD)
        {
            if (playerController == null) return;

            Vector3 spawnPos = new Vector3(bCenter.x, 0.0f, bCenter.z - bD * 0.5f - 1.5f);

            if (buildingData.player_spawn?.position_m?.Length >= 3)
            {
                float spx = buildingData.player_spawn.position_m[0] * modelScaleMultiplier;
                float spz = buildingData.player_spawn.position_m[2] * modelScaleMultiplier;
                if (spz > 0.1f)
                    spawnPos = new Vector3(spx, 0.0f, spz);
                else
                    spawnPos = new Vector3(spx, 0.0f, bCenter.z - bD * 0.5f - 1.5f);
            }

            var cc = playerController.GetComponent<CharacterController>();
            if (cc != null) cc.enabled = false;

            playerController.transform.position = spawnPos;

            Vector3 lookTarget = new Vector3(bCenter.x, 0.0f, bCenter.z);
            Vector3 dir = (lookTarget - spawnPos);
            dir.y = 0f;
            if (dir != Vector3.zero)
                playerController.transform.rotation = Quaternion.LookRotation(dir, Vector3.up);

            if (cc != null) cc.enabled = true;

            if (hudController != null)
            {
                hudController.currentRoomName = "Exterior — South Entry";
                hudController.currentRoomArea = 0f;
            }

            Debug.Log($"[BuildingLoader] Player positioned at {spawnPos}, facing house interior.");
        }

        // ══════════════════════════════════════════════════════
        //  UPDATE — ceiling toggle + room detection
        // ══════════════════════════════════════════════════════
        private void Update()
        {
            bool toggleCeiling = false;
#if ENABLE_INPUT_SYSTEM
            if (Keyboard.current != null)
                toggleCeiling = Keyboard.current.cKey.wasPressedThisFrame;
#else
            toggleCeiling = Input.GetKeyDown(KeyCode.C);
#endif
            if (toggleCeiling && ceilingContainer != null)
            {
                showCeiling = !showCeiling;
                ceilingContainer.SetActive(showCeiling);
            }

            if (!isLoading && playerController != null && hudController != null &&
                buildingData?.rooms != null)
            {
                Vector3 p    = playerController.transform.position;
                float imgX   = (p.x / modelScaleMultiplier) * pxPerM;
                float imgY   = imgHeight - (p.z / modelScaleMultiplier) * pxPerM;
                bool inside  = false;

                foreach (var r in buildingData.rooms)
                {
                    if (r.polygon_flat != null && r.polygon_flat.Length >= 6)
                    {
                        if (PointInPolygonFlat(imgX, imgY, r.polygon_flat))
                        {
                            hudController.currentRoomName = r.label;
                            hudController.currentRoomArea = r.area_m2 * (modelScaleMultiplier * modelScaleMultiplier);
                            inside = true;
                            break;
                        }
                    }
                }
                if (!inside)
                {
                    hudController.currentRoomName = "Exterior";
                    hudController.currentRoomArea = 0f;
                }
            }
        }

        // ══════════════════════════════════════════════════════
        //  PROCEDURAL BOX MESH
        // ══════════════════════════════════════════════════════
        private static Mesh BuildBoxMesh(float width, float height, float depth)
        {
            Vector3 h2 = new Vector3(width * 0.5f, height * 0.5f, depth * 0.5f);

            Vector3[] v = new Vector3[24];
            Vector3[] n = new Vector3[24];
            Vector2[] u = new Vector2[24];
            int[]     t = new int[36];

            // +X right face
            v[ 0] = new Vector3( h2.x, -h2.y, -h2.z); v[ 1] = new Vector3( h2.x,  h2.y, -h2.z);
            v[ 2] = new Vector3( h2.x,  h2.y,  h2.z); v[ 3] = new Vector3( h2.x, -h2.y,  h2.z);
            for (int i= 0;i< 4;i++) n[i]=Vector3.right;
            // -X left face
            v[ 4] = new Vector3(-h2.x, -h2.y,  h2.z); v[ 5] = new Vector3(-h2.x,  h2.y,  h2.z);
            v[ 6] = new Vector3(-h2.x,  h2.y, -h2.z); v[ 7] = new Vector3(-h2.x, -h2.y, -h2.z);
            for (int i= 4;i< 8;i++) n[i]=Vector3.left;
            // +Y top face
            v[ 8] = new Vector3(-h2.x,  h2.y, -h2.z); v[ 9] = new Vector3(-h2.x,  h2.y,  h2.z);
            v[10] = new Vector3( h2.x,  h2.y,  h2.z); v[11] = new Vector3( h2.x,  h2.y, -h2.z);
            for (int i= 8;i<12;i++) n[i]=Vector3.up;
            // -Y bottom face
            v[12] = new Vector3(-h2.x, -h2.y,  h2.z); v[13] = new Vector3(-h2.x, -h2.y, -h2.z);
            v[14] = new Vector3( h2.x, -h2.y, -h2.z); v[15] = new Vector3( h2.x, -h2.y,  h2.z);
            for (int i=12;i<16;i++) n[i]=Vector3.down;
            // +Z front face
            v[16] = new Vector3(-h2.x, -h2.y,  h2.z); v[17] = new Vector3( h2.x, -h2.y,  h2.z);
            v[18] = new Vector3( h2.x,  h2.y,  h2.z); v[19] = new Vector3(-h2.x,  h2.y,  h2.z);
            for (int i=16;i<20;i++) n[i]=Vector3.forward;
            // -Z back face
            v[20] = new Vector3( h2.x, -h2.y, -h2.z); v[21] = new Vector3(-h2.x, -h2.y, -h2.z);
            v[22] = new Vector3(-h2.x,  h2.y, -h2.z); v[23] = new Vector3( h2.x,  h2.y, -h2.z);
            for (int i=20;i<24;i++) n[i]=Vector3.back;

            Vector2[] faceUVs = { Vector2.zero, Vector2.up, Vector2.one, Vector2.right };
            for (int f = 0; f < 6; f++)
                for (int k = 0; k < 4; k++)
                    u[f * 4 + k] = faceUVs[k];

            for (int f = 0; f < 6; f++)
            {
                int b = f * 4, ti = f * 6;
                t[ti+0]=b; t[ti+1]=b+1; t[ti+2]=b+2;
                t[ti+3]=b; t[ti+4]=b+2; t[ti+5]=b+3;
            }

            var mesh = new Mesh();
            mesh.name = "ProcBox";
            mesh.SetVertices(v);
            mesh.SetNormals(n);
            mesh.SetUVs(0, u);
            mesh.SetTriangles(t, 0);
            mesh.RecalculateBounds();
            return mesh;
        }

        // ══════════════════════════════════════════════════════
        //  HELPERS
        // ══════════════════════════════════════════════════════
        private static GameObject SpawnCube(string name, Transform parent, Vector3 localPos, Vector3 scale, Material mat)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            if (parent != null)
            {
                go.transform.SetParent(parent, false);
            }
            go.transform.localPosition = localPos;
            go.transform.localRotation = Quaternion.identity;
            go.transform.localScale    = scale;
            if (mat != null) go.GetComponent<Renderer>().sharedMaterial = mat;
            return go;
        }

        private void ClearBuilding()
        {
            if (generatedBuildingRoot == null) return;

            var children = new List<GameObject>();
            foreach (Transform child in generatedBuildingRoot)
            {
                if (child != null) children.Add(child.gameObject);
            }

            foreach (var go in children)
            {
                if (go != null)
                {
                    if (Application.isPlaying)
                        Destroy(go);
                    else
                        DestroyImmediate(go);
                }
            }
        }

        private static bool PointInPolygonFlat(float x, float y, float[] flat)
        {
            int n = flat.Length / 2;
            bool inside = false;
            int j = n - 1;
            for (int i = 0; i < n; i++)
            {
                float pxi = flat[i*2], pyi = flat[i*2+1];
                float pxj = flat[j*2], pyj = flat[j*2+1];
                if ((pyi < y && pyj >= y || pyj < y && pyi >= y) && (pxi <= x || pxj <= x))
                    if (pxi + (y - pyi) / (pyj - pyi) * (pxj - pxi) < x)
                        inside = !inside;
                j = i;
            }
            return inside;
        }

        private BuildingJson GenerateDefaultBuildingJson()
        {
            var b = new BuildingJson();
            b.coordinate_system = new CoordinateSystemJson { px_per_m = 100f };
            b.building = new BuildingParametersJson { width_m = 12f, depth_m = 10f, wall_height_m = 2.8f, floor_height_m = 0.1f };
            b.player_spawn = new PlayerSpawnJson { position_m = new float[] { 6f, 0.9f, 1.5f } };

            b.walls = new WallJson[]
            {
                new WallJson { id="w1", start=new float[]{0, 0}, end=new float[]{1200, 0}, wall_type="outer", thickness_px=14 },
                new WallJson { id="w2", start=new float[]{1200, 0}, end=new float[]{1200, 1000}, wall_type="outer", thickness_px=14 },
                new WallJson { id="w3", start=new float[]{1200, 1000}, end=new float[]{0, 1000}, wall_type="outer", thickness_px=14 },
                new WallJson { id="w4", start=new float[]{0, 1000}, end=new float[]{0, 0}, wall_type="outer", thickness_px=14 },
                new WallJson { id="w5", start=new float[]{600, 0}, end=new float[]{600, 1000}, wall_type="internal", thickness_px=10 },
                new WallJson { id="w6", start=new float[]{0, 500}, end=new float[]{1200, 500}, wall_type="internal", thickness_px=10 },
            };

            b.rooms = new RoomJson[]
            {
                new RoomJson { id="r1", room_id=1, label="Living Room", area_m2=30f, polygon_flat=new float[]{0,0, 600,0, 600,500, 0,500} },
                new RoomJson { id="r2", room_id=2, label="Master Bedroom", area_m2=30f, polygon_flat=new float[]{600,0, 1200,0, 1200,500, 600,500} },
                new RoomJson { id="r3", room_id=3, label="Kitchen", area_m2=30f, polygon_flat=new float[]{0,500, 600,500, 600,1000, 0,1000} },
                new RoomJson { id="r4", room_id=4, label="Bathroom", area_m2=30f, polygon_flat=new float[]{600,500, 1200,500, 1200,1000, 600,1000} },
            };

            b.doors = new DoorJson[]
            {
                new DoorJson { id="d1", center=new float[]{600, 250}, width_px=60, height_px=120, angle_deg=90 },
                new DoorJson { id="d2", center=new float[]{300, 500}, width_px=60, height_px=120, angle_deg=0 },
                new DoorJson { id="d3", center=new float[]{900, 500}, width_px=60, height_px=120, angle_deg=0 },
            };

            b.windows = new WindowJson[]
            {
                new WindowJson { id="win1", center=new float[]{300, 0}, width_px=80, height_px=100 },
                new WindowJson { id="win2", center=new float[]{900, 0}, width_px=80, height_px=100 },
                new WindowJson { id="win3", center=new float[]{300, 1000}, width_px=80, height_px=100 },
                new WindowJson { id="win4", center=new float[]{900, 1000}, width_px=80, height_px=100 },
            };

            return b;
        }

        private void ShowLoadingHUD(string msg, float progress)
        {
            if (hudController != null)
            {
                hudController.isBuildingLoading = true;
                hudController.currentRoomName   = msg;
                hudController.loadingProgress   = progress;
            }
            Debug.Log($"[BuildingLoader] {Mathf.RoundToInt(progress * 100)}%  {msg}");
        }

        private void HideLoadingHUD()
        {
            if (hudController != null)
            {
                hudController.isBuildingLoading = false;
                hudController.currentRoomName   = "Exterior — South Entry";
                hudController.currentRoomArea   = 0f;
                hudController.loadingProgress   = 1f;
            }
        }
    }
}
