using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

namespace BlueprintX.Runtime
{
    [RequireComponent(typeof(CharacterController))]
    public class FPSController : MonoBehaviour
    {
        [Header("Movement Settings")]
        public float walkSpeed = 3.5f;
        public float sprintSpeed = 6.0f;
        public float jumpHeight = 1.2f;
        public float gravity = -9.81f;

        [Header("Look Settings")]
        public float mouseSensitivity = 2.0f;
        public float upLookLimit = 80.0f;
        public float downLookLimit = -80.0f;

        [Header("References")]
        public Camera playerCamera;

        private CharacterController characterController;
        private Vector3 velocity;
        private float verticalRotation = 0.0f;
        private bool isGrounded;

        private void Start()
        {
            characterController = GetComponent<CharacterController>();
            if (playerCamera == null)
            {
                playerCamera = GetComponentInChildren<Camera>();
            }

            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        private void Update()
        {
            HandleMouseLook();
            HandleMovement();
        }

        private void HandleMouseLook()
        {
            float mouseX = 0f;
            float mouseY = 0f;

#if ENABLE_INPUT_SYSTEM
            if (Mouse.current != null)
            {
                Vector2 delta = Mouse.current.delta.ReadValue();
                mouseX = delta.x * 0.1f * mouseSensitivity;
                mouseY = delta.y * 0.1f * mouseSensitivity;
            }
            else
            {
                mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
                mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;
            }
#else
            mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;
            mouseY = Input.GetAxis("Mouse Y") * mouseSensitivity;
#endif

            verticalRotation -= mouseY;
            verticalRotation = Mathf.Clamp(verticalRotation, downLookLimit, upLookLimit);

            if (playerCamera != null)
            {
                playerCamera.transform.localRotation = Quaternion.Euler(verticalRotation, 0.0f, 0.0f);
            }

            transform.Rotate(Vector3.up * mouseX);
        }

        private void HandleMovement()
        {
            isGrounded = characterController.isGrounded;
            if (isGrounded && velocity.y < 0)
            {
                velocity.y = -2.0f;
            }

            float moveX = 0f;
            float moveZ = 0f;
            bool isSprinting = false;
            bool jumpPressed = false;

#if ENABLE_INPUT_SYSTEM
            if (Keyboard.current != null)
            {
                if (Keyboard.current.wKey.isPressed) moveZ += 1f;
                if (Keyboard.current.sKey.isPressed) moveZ -= 1f;
                if (Keyboard.current.dKey.isPressed) moveX += 1f;
                if (Keyboard.current.aKey.isPressed) moveX -= 1f;

                isSprinting = Keyboard.current.leftShiftKey.isPressed;
                jumpPressed = Keyboard.current.spaceKey.wasPressedThisFrame;
            }
            if (moveX == 0f && moveZ == 0f)
            {
                moveX = Input.GetAxis("Horizontal");
                moveZ = Input.GetAxis("Vertical");
                if (!isSprinting) isSprinting = Input.GetKey(KeyCode.LeftShift);
                if (!jumpPressed) jumpPressed = Input.GetButtonDown("Jump");
            }
#else
            moveX = Input.GetAxis("Horizontal");
            moveZ = Input.GetAxis("Vertical");
            isSprinting = Input.GetKey(KeyCode.LeftShift);
            jumpPressed = Input.GetButtonDown("Jump");
#endif

            float currentSpeed = isSprinting ? sprintSpeed : walkSpeed;
            Vector3 move = transform.right * moveX + transform.forward * moveZ;
            characterController.Move(move.normalized * currentSpeed * Time.deltaTime);

            if (jumpPressed && isGrounded)
            {
                velocity.y = Mathf.Sqrt(jumpHeight * -2.0f * gravity);
            }

            velocity.y += gravity * Time.deltaTime;
            characterController.Move(velocity * Time.deltaTime);
        }
    }
}
