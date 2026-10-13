using System.Collections.Generic;
using UnityEngine;

namespace BlueprintX.Runtime
{
    public class EvacuationManager : MonoBehaviour
    {
        [Header("References")]
        public BuildingLoader buildingLoader;

        [Header("Visualization")]
        public bool showTopology = true;
        public Color nodeColor = Color.cyan;
        public Color edgeColor = Color.green;

        private void OnDrawGizmos()
        {
            if (!showTopology || buildingLoader == null || buildingLoader.buildingData == null) return;
            var topo = buildingLoader.buildingData.topology;
            if (topo == null || topo.nodes == null) return;

            float pxPerM = (buildingLoader.buildingData.coordinate_system != null && buildingLoader.buildingData.coordinate_system.px_per_m > 0)
                ? buildingLoader.buildingData.coordinate_system.px_per_m : 100f;
            float imgHeight = (buildingLoader.buildingData.building != null) ? buildingLoader.buildingData.building.depth_m * pxPerM : 700f;

            Dictionary<int, Vector3> nodePositions = new Dictionary<int, Vector3>();

            Gizmos.color = nodeColor;
            foreach (var node in topo.nodes)
            {
                float wx = node.centroid[0] / pxPerM;
                float wy = 1.0f;
                float wz = (imgHeight - node.centroid[1]) / pxPerM;
                Vector3 pos = new Vector3(wx, wy, wz);
                nodePositions[node.room_id] = pos;
                Gizmos.DrawSphere(pos, 0.35f);
            }

            if (topo.edges != null)
            {
                Gizmos.color = edgeColor;
                foreach (var edge in topo.edges)
                {
                    if (nodePositions.ContainsKey(edge.source) && nodePositions.ContainsKey(edge.target))
                    {
                        Gizmos.DrawLine(nodePositions[edge.source], nodePositions[edge.target]);
                    }
                }
            }
        }
    }
}
