using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace BlueprintX.Runtime
{
    public class PlayerInteraction : MonoBehaviour
    {
        [Header("Settings")]
        public float interactDistance = 3.5f;
        public LayerMask interactableLayer = ~0;

        [Header("References")]
        public Camera playerCamera;

        [Header("Hover Info")]
        public string hoverPrompt = "";

        private void Start()
        {
            if (playerCamera == null)
            {
                playerCamera = GetComponentInChildren<Camera>();
            }
        }

        private void Update()
        {
            UpdateHover();

            bool interactPressed = false;
#if ENABLE_INPUT_SYSTEM
            if (Keyboard.current != null)
            {
                interactPressed = Keyboard.current.eKey.wasPressedThisFrame;
            }
#else
            interactPressed = Input.GetKeyDown(KeyCode.E);
#endif

            if (interactPressed)
            {
                TryInteract();
            }
        }

        private void UpdateHover()
        {
            if (playerCamera == null) return;

            Ray ray = new Ray(playerCamera.transform.position, playerCamera.transform.forward);
            if (Physics.Raycast(ray, out RaycastHit hit, interactDistance, interactableLayer))
            {
                DoorController door = hit.collider.GetComponentInParent<DoorController>();
                if (door != null)
                {
                    hoverPrompt = door.isOpen ? "Press E to Close Door" : "Press E to Open Door";
                    return;
                }
            }
            hoverPrompt = "";
        }

        private void TryInteract()
        {
            if (playerCamera == null) return;

            Ray ray = new Ray(playerCamera.transform.position, playerCamera.transform.forward);
            if (Physics.Raycast(ray, out RaycastHit hit, interactDistance, interactableLayer))
            {
                DoorController door = hit.collider.GetComponentInParent<DoorController>();
                if (door != null)
                {
                    door.ToggleDoor(playerCamera.transform);
                }
            }
        }
    }
}
