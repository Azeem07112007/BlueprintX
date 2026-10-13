using System;
using System.Collections.Generic;
using UnityEngine;

namespace BlueprintX.Runtime
{
    [Serializable]
    public class CoordinateSystemJson
    {
        public string source = "image";
        public string world_axis = "Y_UP";
        public string image_axis = "X_RIGHT_Y_DOWN";
        public float[] origin_offset_m;
        public float px_per_m = 100f;
    }

    [Serializable]
    public class BuildingParametersJson
    {
        public float width_m = 10f;
        public float depth_m = 10f;
        public float wall_height_m = 2.8f;
        public float floor_height_m = 0.1f;
    }

    [Serializable]
    public class PlayerSpawnJson
    {
        public string room_id;
        public string room_label;
        public float[] position_m = new float[3] { 0f, 0.9f, 0f };
        public float rotation_deg;
    }

    [Serializable]
    public class WallJson
    {
        public string id;
        public float[] start = new float[2];
        public float[] end = new float[2];
        public float length_px;
        public float thickness_px = 8f;
        public float angle_deg;
        public string wall_type = "internal";
        public float confidence = 1f;
    }

    [Serializable]
    public class DoorJson
    {
        public string id;
        public float[] center = new float[2];
        public float width_px = 60f;
        public float height_px = 120f;
        public float angle_deg;
        public bool has_arc;
        public string source = "gap";
        public float confidence = 1f;
    }

    [Serializable]
    public class WindowJson
    {
        public string id;
        public float[] center = new float[2];
        public float width_px = 80f;
        public float height_px = 100f;
        public float angle_deg;
        public string source;
        public float confidence;
    }

    [Serializable]
    public class RoomJson
    {
        public string id;
        public int room_id;
        public string label;
        public float[] centroid = new float[2];
        public float area_px;
        public float area_m2;
        public float[] polygon_flat;
        [System.NonSerialized]
        public float[][] polygon_points;
        public int[] adjacent_room_ids;
        public string[] door_ids;
        public string[] window_ids;
    }

    [Serializable]
    public class TopologyNodeJson
    {
        public int room_id;
        public float[] centroid = new float[2];
        public string label;
    }

    [Serializable]
    public class TopologyEdgeJson
    {
        public int source;
        public int target;
        public string type;
        public float[] door_center;
    }

    [Serializable]
    public class TopologyJson
    {
        public TopologyNodeJson[] nodes;
        public TopologyEdgeJson[] edges;
    }

    [Serializable]
    public class BuildingJson
    {
        public string schema_version = "2.1.0";
        public int milestone = 5;
        public string units = "meters";
        public CoordinateSystemJson coordinate_system;
        public BuildingParametersJson building;
        public PlayerSpawnJson player_spawn;
        public WallJson[] walls;
        public DoorJson[] doors;
        public WindowJson[] windows;
        public RoomJson[] rooms;
        public TopologyJson topology;
    }
}
