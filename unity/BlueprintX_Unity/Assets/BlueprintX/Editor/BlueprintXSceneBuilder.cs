using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using BlueprintX.Runtime;

namespace BlueprintX.Editor
{
    public static class BlueprintXSceneBuilder
    {
        private const string ScenePath = "Assets/BlueprintX/Scenes/BlueprintX_Demo.unity";
        private const string BuildDir1 = "D:/BlueprintX/unity/build";
        private const string ExePath1  = "D:/BlueprintX/unity/build/BlueprintX_3D.exe";
        private const string BuildDir2 = "Build";
        private const string ExePath2  = "Build/BlueprintX_3D.exe";

        [MenuItem("BlueprintX/Setup Demo Scene")]
        public static void SetupDemoScene()
        {
            if (EditorApplication.isPlaying)
            {
                Debug.LogWarning("[BlueprintXSceneBuilder] Unity is in Play mode. Stopping Play mode...");
                EditorApplication.isPlaying = false;
                return;
            }

            Debug.Log("[BlueprintXSceneBuilder] Setting up BlueprintX_Demo scene with URP materials & outdoor environment...");

            // 0. Ensure URP Material template exists in Resources folder so standalone builds never turn pink
            EnsureUrpResourceMaterial();

            // 1. Create fresh scene
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            // 2. Directional Sunlight
            var lightObj = new GameObject("Directional Light");
            var light = lightObj.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 0.85f;
            light.color = new Color(1f, 0.97f, 0.92f);
            light.shadows = LightShadows.Soft;
            light.shadowStrength = 0.55f;
            lightObj.transform.rotation = Quaternion.Euler(45f, -35f, 0f);

            // 2b. Balanced Ambient Lighting
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.42f, 0.44f, 0.46f);

            // 3. Player GameObject (Positioned outside facing the house)
            var playerObj = new GameObject("Player");
            playerObj.tag = "Player";
            Vector3 spawnPos = new Vector3(6.0f, 0.0f, -5.5f);
            playerObj.transform.position = spawnPos;
            playerObj.transform.rotation = Quaternion.LookRotation(new Vector3(6.0f, 1.4f, 5.0f) - spawnPos);

            var charController = playerObj.AddComponent<CharacterController>();
            charController.height = 1.8f;
            charController.radius = 0.35f;
            charController.center = new Vector3(0f, 0.9f, 0f);

            var fps = playerObj.AddComponent<FPSController>();
            var interaction = playerObj.AddComponent<PlayerInteraction>();

            // Child Camera
            var camObj = new GameObject("PlayerCamera");
            camObj.tag = "MainCamera";
            camObj.transform.parent = playerObj.transform;
            camObj.transform.localPosition = new Vector3(0f, 1.6f, 0f);
            camObj.transform.localRotation = Quaternion.identity;

            var cam = camObj.AddComponent<Camera>();
            cam.nearClipPlane = 0.1f;
            cam.farClipPlane = 1000f;
            cam.allowHDR = false;
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.42f, 0.65f, 0.90f); // Vibrant Sky Blue
            camObj.AddComponent<AudioListener>();

            fps.playerCamera = cam;
            interaction.playerCamera = cam;

            // 4. GeneratedBuilding Root
            var buildingRoot = new GameObject("GeneratedBuilding");
            buildingRoot.transform.position = Vector3.zero;

            // 5. HUD
            var hudObj = new GameObject("HUD");
            var hud = hudObj.AddComponent<HUDController>();
            hud.playerInteraction = interaction;

            // 6. BuildingLoader
            var loaderObj = new GameObject("BuildingLoader");
            var loader = loaderObj.AddComponent<BuildingLoader>();
            loader.playerController = fps;
            loader.hudController = hud;
            loader.generatedBuildingRoot = buildingRoot.transform;
            loader.modelScaleMultiplier = 3.0f;        // full walking-scale rooms
            loader.wallHeight = 3.2f;                  // storey height metres
            loader.minOuterWallThickness = 0.30f;
            loader.minInternalWallThickness = 0.20f;
            loader.showCeiling = true;
            hud.buildingLoader = loader;

            // 7. Simulation / EvacuationManager
            var simObj = new GameObject("EvacuationManager");
            var evac = simObj.AddComponent<EvacuationManager>();
            evac.buildingLoader = loader;

            // Ensure directory exists
            string dir = Path.GetDirectoryName(ScenePath);
            if (!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
            }

            // Save Scene
            bool saved = EditorSceneManager.SaveScene(scene, ScenePath);
            Debug.Log($"[BlueprintXSceneBuilder] Scene saved to {ScenePath}: {saved}");

            // Configure Build Settings
            var scenes = new EditorBuildSettingsScene[]
            {
                new EditorBuildSettingsScene(ScenePath, true)
            };
            EditorBuildSettings.scenes = scenes;
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
        }

        private static void EnsureUrpResourceMaterial()
        {
            try
            {
                string resourcesDir = "Assets/BlueprintX/Resources";
                if (!Directory.Exists(resourcesDir))
                {
                    Directory.CreateDirectory(resourcesDir);
                }

                string matPath = Path.Combine(resourcesDir, "URPMaterial.mat");
                Material urpMat = AssetDatabase.LoadAssetAtPath<Material>(matPath);
                if (urpMat == null)
                {
                    Shader urpShader = Shader.Find("Universal Render Pipeline/Lit");
                    if (urpShader == null) urpShader = Shader.Find("Universal Render Pipeline/Simple Lit");
                    if (urpShader == null) urpShader = Shader.Find("Universal Render Pipeline/Unlit");
                    if (urpShader == null) urpShader = Shader.Find("Standard");

                    if (urpShader != null)
                    {
                        urpMat = new Material(urpShader);
                        AssetDatabase.CreateAsset(urpMat, matPath);
                        Debug.Log($"[BlueprintXSceneBuilder] Successfully created URP Material asset at {matPath}");
                    }
                }

                string glassMatPath = Path.Combine(resourcesDir, "GlassMaterial.mat");
                Material glassMat = AssetDatabase.LoadAssetAtPath<Material>(glassMatPath);
                if (glassMat == null)
                {
                    Shader s = Shader.Find("Standard");
                    if (s == null) s = Shader.Find("Unlit/Transparent");
                    if (s == null) s = Shader.Find("Universal Render Pipeline/Unlit");

                    if (s != null)
                    {
                        glassMat = new Material(s);
                        Color glassColor = new Color(0.65f, 0.85f, 0.96f, 0.18f);
                        glassMat.SetFloat("_Mode", 3.0f);
                        glassMat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.One);
                        glassMat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                        glassMat.SetInt("_ZWrite", 0);
                        glassMat.DisableKeyword("_ALPHATEST_ON");
                        glassMat.DisableKeyword("_ALPHABLEND_ON");
                        glassMat.EnableKeyword("_ALPHAPREMULTIPLY_ON");
                        glassMat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
                        glassMat.SetOverrideTag("RenderType", "Transparent");
                        glassMat.color = glassColor;
                        if (glassMat.HasProperty("_Color")) glassMat.SetColor("_Color", glassColor);
                        if (glassMat.HasProperty("_BaseColor")) glassMat.SetColor("_BaseColor", glassColor);
                        glassMat.SetFloat("_Surface", 1.0f);
                        glassMat.SetFloat("_Blend", 0.0f);
                        glassMat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                        AssetDatabase.CreateAsset(glassMat, glassMatPath);
                        Debug.Log($"[BlueprintXSceneBuilder] Successfully created Glass Material asset at {glassMatPath}");
                    }
                }
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[BlueprintXSceneBuilder] Material creation notice: {ex.Message}");
            }
        }

        [MenuItem("BlueprintX/Build Standalone Player")]
        public static void BuildStandalone()
        {
            SetupDemoScene();

            string targetBuildDir = BuildDir1;
            string targetExePath  = ExePath1;

            if (!Directory.Exists(targetBuildDir))
            {
                Directory.CreateDirectory(targetBuildDir);
            }

            Debug.Log($"[BlueprintXSceneBuilder] Building standalone player to {targetExePath}...");

            BuildPlayerOptions options = new BuildPlayerOptions
            {
                scenes = new[] { ScenePath },
                locationPathName = targetExePath,
                target = BuildTarget.StandaloneWindows64,
                options = BuildOptions.None
            };

            var report = BuildPipeline.BuildPlayer(options);
            Debug.Log($"[BlueprintXSceneBuilder] Build completed with result: {report.summary.result} ({report.summary.totalErrors} errors, size: {report.summary.totalSize} bytes)");

            // Copy to local project Build directory as well
            try
            {
                if (!Directory.Exists(BuildDir2)) Directory.CreateDirectory(BuildDir2);
                if (File.Exists(targetExePath)) File.Copy(targetExePath, ExePath2, true);
            }
            catch (System.Exception ex)
            {
                Debug.LogWarning($"[BlueprintXSceneBuilder] Secondary copy warning: {ex.Message}");
            }

            // Ensure StreamingAssets are copied into build data folder
            string srcStreaming = "Assets/StreamingAssets";
            string dstStreaming = Path.Combine(targetBuildDir, "BlueprintX_3D_Data", "StreamingAssets");
            if (Directory.Exists(srcStreaming))
            {
                if (!Directory.Exists(dstStreaming)) Directory.CreateDirectory(dstStreaming);
                foreach (var file in Directory.GetFiles(srcStreaming))
                {
                    if (!file.EndsWith(".meta"))
                    {
                        File.Copy(file, Path.Combine(dstStreaming, Path.GetFileName(file)), true);
                    }
                }
                Debug.Log($"[BlueprintXSceneBuilder] Copied fresh StreamingAssets to {dstStreaming}");
            }
        }
    }
}
