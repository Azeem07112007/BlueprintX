using System.Collections;
using UnityEngine;

namespace BlueprintX.Runtime
{
    public class DoorController : MonoBehaviour
    {
        public string doorId = "";
        public bool isOpen = false;
        public float openAngle = 90f;
        public float openSpeed = 5f;

        private Quaternion closedRotation;
        private Quaternion targetRotation;
        private Coroutine activeAnimation;

        private void Awake()
        {
            closedRotation = transform.localRotation;
            targetRotation = closedRotation;
        }

        private void Start()
        {
            closedRotation = transform.localRotation;
            targetRotation = closedRotation;
        }

        public void ToggleDoor(Transform interacter = null)
        {
            isOpen = !isOpen;

            if (isOpen)
            {
                float swingSign = 1f;
                if (interacter != null)
                {
                    // Compute player direction relative to door frame forward vector
                    // Positive dot = player in front -> push AWAY into house (+openAngle / +90 deg)
                    // Negative dot = player behind -> push AWAY to exterior (-openAngle / -90 deg)
                    Vector3 doorForward = transform.forward;
                    Vector3 playerDir = (interacter.position - transform.position).normalized;
                    float dot = Vector3.Dot(playerDir, doorForward);
                    swingSign = (dot >= 0f) ? 1f : -1f;
                }
                targetRotation = closedRotation * Quaternion.Euler(0f, openAngle * swingSign, 0f);
            }
            else
            {
                targetRotation = closedRotation;
            }

            if (activeAnimation != null)
            {
                StopCoroutine(activeAnimation);
            }
            activeAnimation = StartCoroutine(AnimateDoor(targetRotation));
        }

        private IEnumerator AnimateDoor(Quaternion targetRot)
        {
            while (Quaternion.Angle(transform.localRotation, targetRot) > 0.5f)
            {
                transform.localRotation = Quaternion.Slerp(transform.localRotation, targetRot, Time.deltaTime * openSpeed);
                yield return null;
            }
            transform.localRotation = targetRot;
        }
    }
}
