using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;

public class RobotController : MonoBehaviour
{
    // naming constraints do not change
    [SerializeField] private WheelCollider FLC;
    [SerializeField] private WheelCollider FRC;
    [SerializeField] private WheelCollider RLC;
    [SerializeField] private WheelCollider RRC;

    [SerializeField] private Transform FLT;
    [SerializeField] private Transform FRT;
    [SerializeField] private Transform RLT;
    [SerializeField] private Transform RRT;

    [SerializeField] private Transform FRS;
    [SerializeField] private Transform L1S;
    [SerializeField] private Transform L2S;
    [SerializeField] private Transform L3S;
    [SerializeField] private Transform R1S;
    [SerializeField] private Transform R2S;
    [SerializeField] private Transform R3S;
    [SerializeField] private Transform ORS;
    [SerializeField] private Transform Down;

    [Header("Movement Parameters")]
    public float motorTorque = 20f;

    public float brakeTorque = 20f;
    public float turnSpeed = 30f;

    [Header("Sensor Parameters")]
    public float sensorRange = 10f;
    public float obstacleDetectionDistance = 7f;
    public string roadMaterial = "MT_Road_01";

    [Header("Performance Tuning")]
    public float finalBrakeForce = -140f;
    public float steeringSmoothing = 5f;
    public float accelerationSmoothing = 2f;

    public float decelerationSmoothing = 10f;

    private float currentSteerAngle = 0f;
    private float currentMotorTorque = 0f;

    private float currentBrakeTorque = 0f;

    private GeneticAlgorithm ga;
    private int individualIndex;
    private float totalReward;
    private List<Vector2> currentIndividual;
    private bool isActive;
    public bool shouldRender = true; // Controls visual updates

    private float totalTorqueReward = 0f;
    private float totalSteeringReward = 0f;

    private bool finishingPointDetected = false;
    private Vector3 lastPosition;

    public float GetCurrentMotorTorque() => currentMotorTorque;
    public float GetCurrentSteerAngle() => currentSteerAngle;

    public float GetSteeringReward() => totalSteeringReward;

    private void Start()
    {
        ga = FindObjectOfType<GeneticAlgorithm>();
    }

    public void InitializeForGA(GeneticAlgorithm geneticAlgorithm, int index)
    {
        ga = geneticAlgorithm;
        individualIndex = index;
        totalSteeringReward = 0f;
        totalTorqueReward = 0f;
        isActive = true;
        ManualReset();
    }

    public void UpdateFitness(bool checkSpeed)
    {
        if (!isActive) return;

        float torqueReward = HandleTorqueRewards(currentMotorTorque);
        float steeringReward = HandleSteeringRewards(currentSteerAngle, currentMotorTorque);
        // Debug.Log($"Motor torque: {currentMotorTorque}, steer: {currentSteerAngle}");
        totalTorqueReward += torqueReward;
        totalSteeringReward += steeringReward;

        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);

        if (IsOutOfTrack() || (checkSpeed && speed <= 0.2f))
        {
            ga.UpdateFitness(individualIndex, totalTorqueReward, totalSteeringReward, true);
            isActive = false;
            GetComponent<Rigidbody>().isKinematic = true;
        }
        else if (HandleFinalCheckpoint())
        {
            ga.UpdateFitness(individualIndex, totalTorqueReward, totalSteeringReward, true);
            isActive = false;
            GetComponent<Rigidbody>().isKinematic = true;
        }
    }

    public float HandleTorqueRewards(float motorTorque)
    {
        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);

        // Debug.Log($"Speed : {speed}");
        float reward = 0f;

        if (speed > 0f)
        {
            reward += speed > 2f ? (speed < 6f ? 1f : -0.3f) : -0.1f;
        }
        else
        {
            reward -= 1f;
        }

        if (IsOutOfTrack())
        {
            reward -= 10f;
        }

        return reward;
    }

    public float GetSpeed()
    {
        return Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);
    }

    public float HandleSteeringRewards(float steeringAngle, float torque)
    {
        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);
        float reward = 0f;
        int road = GetRoad();

        if (speed > 0f)
        {
            reward += speed > 2f ? (speed < 6f ? 1f : -0.3f) : -1f;
        }
        else
        {
            reward -= 1f;
        }

        if (road == 1)
        {
            if (steeringAngle > 5f && steeringAngle < 30f)
            {
                // Debug.Log("Turning rewards adding");
                reward += steeringAngle > 5f ? 1f : 0f;
                reward += steeringAngle > 10f ? 2f : 0f;
                reward += steeringAngle > 15f ? 3f : 0f;
            }
            else
                reward += -5f;
        }
        else if (road == 2)
        {
            if (steeringAngle < -5f && steeringAngle > -15f)
            {
                // Debug.Log("Turning rewards adding");
                reward += steeringAngle < -5f ? 1f : 0f;
                reward += steeringAngle < -10f ? 2f : 0f;
                reward += steeringAngle < -15f ? 3f : 0f;
            }
            else
                reward += -5f;
        }

        if (road == 0 && steeringAngle > -10f && steeringAngle < 10f)
            reward += 1f;
        else
            reward += -1f;

        (float, string) ors = CheckOrientationSensor();
        if (ors.Item1 <= -2f)
        {
            if (torque >= 250f)
                reward += 3f;
            else if (torque <= 150f)
                reward += -1f;
        }
        reward += HandleEdgeDetection();

        // ✅ ➕ Add reward for distance covered since last frame
        float deltaDistance = Vector3.Distance(transform.position, lastPosition);
        reward += deltaDistance * 100f; // 🔁 2f is the weight — adjust as needed
        // Debug.Log($"Last position: {lastPosition}, current position: {transform.position}, deltaDistance: {deltaDistance}");

        // Update last position
        lastPosition = transform.position;

        return reward;
    }

    public void SetIndividual(List<Vector2> individual)
    {
        currentIndividual = individual; ;
        totalSteeringReward = 0f;
        totalTorqueReward = 0f;
        isActive = true;
        // ✅ Reset steering and torque for fresh generation
        currentMotorTorque = 0f;
        currentSteerAngle = 0f;
    }

    // private void FixedUpdate()
    // {
    //     var sensorReadings = GetSensorData();
    //     HandleNavigation(sensorReadings);
    //     UpdateWheelTransforms();
    //     // Debug.Log("Time");
    // }

    private float HandleEdgeDetection()
    {
        var sensorReadings = GetSensorData();

        if (sensorReadings["Left3"].Item2.StartsWith("ED") || sensorReadings["Right3"].Item2.StartsWith("ED") || sensorReadings["Left3"].Item2.StartsWith("Plane") || sensorReadings["Right3"].Item2.StartsWith("Plane"))
        {
            // Debug.Log("Close to edge");
            return -3f;

        }

        return 5f;
    }

    public void ManualReset()
    {
        
        // transform.localPosition = new Vector3(34.56854f, 23.92629f, -243.2978f); // first position for GA
        // transform.rotation = Quaternion.Euler(0f, 177.441f, -0.001f);

        transform.localPosition = new Vector3(-94.5086f, 39.55402f, -303.3212f); // seond position for GA
        transform.rotation = Quaternion.Euler(-0.31f, 360.243f, 3.421f);

        lastPosition = transform.position;
        Rigidbody rb = GetComponent<Rigidbody>();
        rb.isKinematic = false;
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        // Sleep & wake for clean reset
        rb.Sleep();
        rb.WakeUp();
        ApplyMotorTorque(0f);
        ApplySteering(0f);
        SetSensorOrientations();
    }

    public int GetRoad()
    {
        var sensorReadings = GetSensorData();
        string[] keysToCheck = { "Down", };

        foreach (var key in keysToCheck)
        {
            string hitObject = sensorReadings[key].Item2;
            // Debug.Log($"Hitobject: {hitObject}");

            if (hitObject.Contains("MT_Turn (1)") || hitObject.Contains("MT_Turn (2)") || hitObject.Contains("MT_Turn (8)"))
            {
                return 1;
            }
            else if (hitObject.Contains("MT_Turn (13)") || hitObject.Contains("MT_Turn (14)") || hitObject.Contains("MT_Turn (7)") || hitObject.Contains("MT_Turn (6)") || hitObject.Contains("MT_Turn (5)") || hitObject.Contains("MT_Turn (11)"))
            {
                return 2;
            }
            else if (hitObject.Contains("MT_Turn (15)") || hitObject.Contains("MT_Turn (9)") || hitObject.Contains("MT_Turn (12)") || hitObject.Contains("MT_Turn (10)"))
            {
                return 3;
            }
        }

        return 0;
    }

    // Sensor Data Collection
    public Dictionary<string, (float, string)> GetSensorData()
    {
        return new Dictionary<string, (float, string)>
        {
            { "Front", CheckSensor(FRS, true) },
            { "Left1", CheckSensor(L1S, true) },
            { "Left2", CheckSensor(L2S, true) },
            { "Left3", CheckSensor(L3S, true) },
            { "Right1", CheckSensor(R1S, true) },
            { "Right2", CheckSensor(R2S, true) },
            { "Right3", CheckSensor(R3S, true) },
            { "Down", CheckSensor(Down) },
            { "ORS", CheckOrientationSensor() },
            { "ORSZ", CheckOrientationSensorZ() }
        };
    }

    private void SetSensorOrientations()
    {
        FRS.localRotation = Quaternion.Euler(8, 0, 0);
        L1S.localRotation = Quaternion.Euler(8, -15, 0);
        L2S.localRotation = Quaternion.Euler(8, -35, 0);
        L3S.localRotation = Quaternion.Euler(15, -90, 0);
        R1S.localRotation = Quaternion.Euler(8, 15, 0);
        R2S.localRotation = Quaternion.Euler(8, 35, 0);
        R3S.localRotation = Quaternion.Euler(15, 90, 0);
        Down.localRotation = Quaternion.Euler(90, 0, 0);
    }

    public (float, string) CheckOrientationSensor()
    {
        float xaw = transform.eulerAngles.x;
        float normalizedPitch = (xaw > 180) ? xaw - 360 : xaw;
        return (normalizedPitch, "OrientationX");
    }

    private (float, string) CheckOrientationSensorZ()
    {
        float zaw = transform.eulerAngles.z;
        float normalizedPitch = (zaw > 180) ? zaw - 360 : zaw;
        return (normalizedPitch, "OrientationZ");
    }

    private void HandleNavigation(Dictionary<string, (float, string)> sensorReadings)
    {
        float moveSpeed = motorTorque; // Default speed
        float steerAngle = 0f;




        // Check distances to road edges on both sides
        var leftEdge = sensorReadings["Left3"].Item2.StartsWith("ED") || sensorReadings["Left3"].Item2.StartsWith("Plane") ? sensorReadings["Left3"].Item1 : sensorRange;
        var rightEdge = sensorReadings["Right3"].Item2.StartsWith("ED") || sensorReadings["Right3"].Item2.StartsWith("Plane") ? sensorReadings["Right3"].Item1 : sensorRange;
        float deviation = leftEdge - rightEdge;

        if (sensorReadings["Left3"].Item2.StartsWith("None") && (sensorReadings["Right3"].Item2.StartsWith("MT_Road_01") || sensorReadings["Right3"].Item2.StartsWith("MT_Turn")))
        {
            deviation = sensorReadings["Right3"].Item1 - sensorRange;
            Debug.Log("Special check1");
        }

        if (sensorReadings["Right3"].Item2.StartsWith("None") && (sensorReadings["Left3"].Item2.StartsWith("MT_Road_01") || sensorReadings["Left3"].Item2.StartsWith("MT_Turn")))
        {
            deviation = sensorRange - sensorReadings["Left3"].Item1;
            Debug.Log("Special check2");
        }

        if (sensorReadings["Right3"].Item2.StartsWith("Cube") || sensorReadings["Left3"].Item2.StartsWith("Cube"))
        {
            deviation = 0;
        }

        Debug.Log($"Deviation: {deviation}, LeftEdge: {leftEdge}, LeftEdgeItem: {sensorReadings["Left3"].Item2}, RightEdge: {rightEdge}, RightEdgeItem: {sensorReadings["Right3"].Item2},");
        Debug.Log($"Left1Item: {sensorReadings["Left1"].Item2}, Left2Item: {sensorReadings["Left2"].Item2}");
        Debug.Log($"Right1Item: {sensorReadings["Right1"].Item2}, Right2Item: {sensorReadings["Right2"].Item2},");

        // Adjust steering angle based on deviation
        if (Mathf.Abs(deviation) > 0.5f && !(sensorReadings["Left3"].Item2.StartsWith("CP") && sensorReadings["Right3"].Item2.StartsWith("CP"))) // Adjust threshold as needed for sensitivity
        {
            steerAngle = -deviation * turnSpeed / sensorRange;
            // Debug.Log($"steerAngle from LS3 and RS3: {steerAngle}");
        }

        // Check for turning point using all front sensors except Left3 and Right3
        bool isTurningPointDetected =
            sensorReadings["Front"].Item2.StartsWith("MT_Turn") ||
            sensorReadings["Left1"].Item2.StartsWith("MT_Turn") ||
            sensorReadings["Left2"].Item2.StartsWith("MT_Turn") ||
            sensorReadings["Right1"].Item2.StartsWith("MT_Turn") ||
            sensorReadings["Right2"].Item2.StartsWith("MT_Turn");

        if (isTurningPointDetected)
        {
            // // Debug.Log("Turning point detected. Hard braking applied.");

            // Hard brake by reducing speed aggressively
            moveSpeed = Mathf.Min(moveSpeed - (motorTorque / 5f), motorTorque / 6f); // Gradual but sharp reduction
            // // Debug.Log($"Slowing speed calculated: {moveSpeed}");
        }

        // Log ORS orientation sensor reading
        if (sensorReadings.ContainsKey("ORS"))
        {
            var orsReading = sensorReadings["ORS"];
            float pitch = orsReading.Item1;

            // // Debug.Log($"Pitch: {pitch} ");


            // Adjust speed based on pitch
            if (pitch <= -2f) // Upward slope, mild to steep
            {
                if (!isTurningPointDetected || moveSpeed <= 3.5f)
                {

                    // // Debug.Log($"Uphill detected: Pitch = {pitch}. Increasing torque for acceleration.");
                    // // Debug.Log($"FrontItem: {sensorReadings["Front"].Item2}");
                    moveSpeed += motorTorque * 20f >= 270f ? 270f : motorTorque * 20f; // Boost acceleration
                }

            }
            else if (pitch > 2f) // Downward slope, mild to steep
            {
                if (!isTurningPointDetected)
                {
                    // // Debug.Log($"Downhill detected: Pitch = {pitch}. Applying brake.");
                    moveSpeed = (moveSpeed - (motorTorque * 2f)) <= 20 ? 20 : (moveSpeed - (motorTorque * 2f)); // Apply brake by reducing torque
                }
            }
            else
            {
                // // Debug.Log($"Flat terrain detected: Pitch = {pitch}. Maintaining default torque.");
            }
        }

        bool obstacleDetected = false;

        // Check for obstacles in front and react accordingly
        if (checkObstacle(sensorReadings, "Front") || checkObstacle(sensorReadings, "Left1") || checkObstacle(sensorReadings, "Left2") || checkObstacle(sensorReadings, "Right1") || checkObstacle(sensorReadings, "Right2"))
        {
            // Debug.Log("Obstacle detected");
            bool leftClear = sensorReadings["Left2"].Item1 > obstacleDetectionDistance ? sensorReadings["Left1"].Item1 > obstacleDetectionDistance : false;
            bool rightClear = sensorReadings["Right2"].Item1 > obstacleDetectionDistance ? sensorReadings["Right1"].Item1 > obstacleDetectionDistance : false;

            if (rightClear)
            {
                // steerAngle = isTurningPointDetected ? turnSpeed * 10 : turnSpeed; // Turn right
                if (sensorReadings["Left2"].Item2.StartsWith("Cube") && sensorReadings["Left2"].Item1 < obstacleDetectionDistance - 3 && sensorReadings["Left1"].Item1 > obstacleDetectionDistance)
                {

                    steerAngle = turnSpeed - 20;
                    // Debug.Log($"LF2 detected object turnspeed: {steerAngle}");
                }
                else if (sensorReadings["Left1"].Item2.StartsWith("Cube") && sensorReadings["Left1"].Item1 < obstacleDetectionDistance - 2)
                {
                    steerAngle = turnSpeed - 10;
                    // Debug.Log($"LF1 detected object turnspeed: {steerAngle}");
                }
                else if (!sensorReadings["Left1"].Item2.StartsWith("Cube") && !sensorReadings["Left2"].Item2.StartsWith("Cube"))
                {

                    steerAngle = turnSpeed - 5;
                    // Debug.Log($"Other object detected. turnspeed: {steerAngle}");
                }
                else
                    steerAngle = currentSteerAngle / 2f;
                obstacleDetected = true;

                // Debug.Log($"steerAngle when rightClear: {steerAngle}");
            }
            else if (leftClear)
            {
                // steerAngle = isTurningPointDetected ? -turnSpeed * 10 : -turnSpeed; // Turn left
                if (sensorReadings["Right2"].Item2.StartsWith("Cube") && sensorReadings["Right2"].Item1 < obstacleDetectionDistance - 3 && sensorReadings["Right1"].Item1 > obstacleDetectionDistance)
                {
                    steerAngle = -turnSpeed + 25;
                    Debug.Log($"RF2 detected object turnspeed: {steerAngle}");
                }
                else if (sensorReadings["Right1"].Item2.StartsWith("Cube") && sensorReadings["Right1"].Item1 < obstacleDetectionDistance - 2)
                {
                    steerAngle = -turnSpeed + 15;
                    Debug.Log($"RF1 detected object turnspeed: {steerAngle}");
                }

                else if (!sensorReadings["Right1"].Item2.StartsWith("Cube") && !sensorReadings["Right2"].Item2.StartsWith("Cube"))
                    steerAngle = -turnSpeed + 5;
                else
                    steerAngle = currentSteerAngle / 2f;

                obstacleDetected = true;

                // Debug.Log($"steerAngle when leftClear: {steerAngle}");
            }
            else
                moveSpeed = Mathf.Max(moveSpeed - (motorTorque / 8f), motorTorque / 8f); // Slow down further
        }

        // Debug.Log($"Final Target Speed: {moveSpeed}");

        if (moveSpeed > 200f)
        {
            // Calculate the difference from 200
            float speedExcess = moveSpeed - 200f;

            // Scale the reduction (tweak the factor as needed)
            float reductionFactor = 0.1f; // Adjust this value for sensitivity
            if (steerAngle > 0f)
                steerAngle -= speedExcess * reductionFactor;
            else if (steerAngle < 0f)
                steerAngle += speedExcess * reductionFactor;

            // Optionally clamp the steerAngle to avoid excessive reduction
            steerAngle = Mathf.Clamp(steerAngle, -turnSpeed + 5, turnSpeed - 5); // Replace 'maxSteerAngle' with your max limit
            Debug.Log($"Adjusted steering angle: {steerAngle}");
        }

        if (sensorReadings["Left3"].Item2.StartsWith("ED") && sensorReadings["Right3"].Item2.StartsWith("ED"))
        {
            bool isFrontEmpty = sensorReadings["Left2"].Item2.StartsWith("None") &&
                sensorReadings["Left1"].Item2.StartsWith("None") &&
                sensorReadings["Right1"].Item2.StartsWith("None") &&
                sensorReadings["Front"].Item2.StartsWith("None") &&
                sensorReadings["Right2"].Item2.StartsWith("None");

            if (isFrontEmpty)
            {
                // Get the current speed of the car
                float currentSpeed = moveSpeed; // Assume moveSpeed holds the current speed in units/second

                // Calculate the required deceleration to stop in 2 seconds
                float deceleration = currentSpeed / 2f;

                // Apply braking force
                moveSpeed -= deceleration * Time.deltaTime;

                // Ensure speed doesn't go negative
                moveSpeed = -Mathf.Max(moveSpeed, 0f);

            }

        }

        // if (isTurningPointDetected)
        // {
        bool leftEdgeDetected = false;
        bool righEdgeDetected = false;
        if ((sensorReadings["Left1"].Item2.StartsWith("ED") || sensorReadings["Left2"].Item2.StartsWith("ED") || sensorReadings["Left3"].Item2.StartsWith("ED")) &&
            (!sensorReadings["Right1"].Item2.StartsWith("ED") && !sensorReadings["Right2"].Item2.StartsWith("ED") && !sensorReadings["Right3"].Item2.StartsWith("ED"))
        )
        {
            leftEdgeDetected = true;
            // Debug.Log($"Detection: leftEdge, steeringAngle: {steerAngle}");
        }
        else if (!sensorReadings["Left1"].Item2.StartsWith("None") && !sensorReadings["Left2"].Item2.StartsWith("None") && !sensorReadings["Left3"].Item2.StartsWith("None") &&
            sensorReadings["Right1"].Item2.StartsWith("None") && sensorReadings["Right2"].Item2.StartsWith("None") && sensorReadings["Right3"].Item2.StartsWith("None")
        )
        {
            leftEdgeDetected = true;
        }

        if ((sensorReadings["Right1"].Item2.StartsWith("ED") || sensorReadings["Right2"].Item2.StartsWith("ED") || sensorReadings["Right3"].Item2.StartsWith("ED")) &&
            (!sensorReadings["Left1"].Item2.StartsWith("ED") && !sensorReadings["Left2"].Item2.StartsWith("ED") && !sensorReadings["Left3"].Item2.StartsWith("ED"))
        )
        {
            righEdgeDetected = true;
            // Debug.Log($"Detection: rightEdge, steeringAngle: {steerAngle}");
        }
        else if (!sensorReadings["Right1"].Item2.StartsWith("None") && !sensorReadings["Right2"].Item2.StartsWith("None") && !sensorReadings["Right3"].Item2.StartsWith("None") &&
            sensorReadings["Left1"].Item2.StartsWith("None") && sensorReadings["Left2"].Item2.StartsWith("None") && sensorReadings["Left3"].Item2.StartsWith("None")
        )
        {
            righEdgeDetected = true;
        }

        // Debug.Log($"Detection: steeringAngle: {steerAngle}");

        if ((leftEdgeDetected || righEdgeDetected) && !(leftEdgeDetected && righEdgeDetected) && !obstacleDetected)
        {
            if (leftEdgeDetected)
            {
                if (steerAngle < 5)
                    steerAngle = 5f;
                // Debug.Log($"Detection: leftEdge, steeringAngle: {steerAngle}");

            }
            else
            {
                if (steerAngle > 5f)
                    steerAngle = -5f;
                // Debug.Log($"Detection: rightEdge, steeringAngle: {steerAngle}");
            }
        }


        ApplySteering(steerAngle);
        ApplyMotorTorque(moveSpeed);
    }

    private bool checkFinishingPoint(Dictionary<string, (float, string)> sensorReadings)
    {
        return sensorReadings["Left3"].Item2.StartsWith("CP22") || sensorReadings["Right3"].Item2.StartsWith("CP22");
    }

    private bool checkObstacle(Dictionary<string, (float, string)> sensorReadings, string sensor)
    {
        return sensorReadings[sensor].Item1 < obstacleDetectionDistance && !sensorReadings[sensor].Item2.StartsWith("CP") && !sensorReadings[sensor].Item2.StartsWith("MT_Road") && !sensorReadings[sensor].Item2.StartsWith("MT_Turn");
    }


    private (float, string) CheckSensor(Transform sensor, bool draw = false)
    {
        RaycastHit hit;

        // Exclude 'Robot' layer
        int layerMask = ~LayerMask.GetMask("Robot");

        // ✅ Only draw if this is the Down sensor
        if (draw)
        {
            Debug.DrawRay(sensor.position, sensor.forward * sensorRange, Color.yellow);
        }

        if (Physics.Raycast(sensor.position, sensor.forward, out hit, sensorRange, layerMask))
        {
            return (hit.distance, hit.collider.gameObject.name);
        }
        return (sensorRange, "None");
    }

    // Manual Control Application
    public void ManualApplyControl(float torque, float steering)
    {
        ApplySteering(steering);
        ApplyMotorTorque(torque);
        // Debug.Log($"Torque: {torque}, steering: {currentSteerAngle}, car: {individualIndex}");
        UpdateWheelTransforms();
    }

    // Handle Final Checkpoint
    public bool HandleFinalCheckpoint()
    {
        return HasPassedFinalCheckpoint() && IsStopped();
    }

    private bool HasPassedFinalCheckpoint()
    {
        Collider[] hitColliders = Physics.OverlapSphere(transform.position, 1f);
        foreach (var collider in hitColliders)
        {
            if (collider.gameObject.name.StartsWith("CP22"))
                return true;
        }
        return false;
    }

    private bool IsStopped()
    {
        return GetComponent<Rigidbody>().velocity.magnitude < 0.1f;
    }

    // Check if Out of Track
    public bool IsOutOfTrack()
    {
        var sensorReadings = GetSensorData();
        string hitObject = sensorReadings["Down"].Item2;

        // Debug.Log($"Down hit: {hitObject}");

        if (hitObject.StartsWith("MT_Road") || hitObject.StartsWith("MT_Turn"))
            return false; // ✅ Still on track

        if (hitObject.StartsWith("ED"))
            return true; // ❌ Explicitly off-track

        return true; // ❌ Anything else = unknown = off-track
    }

    private void ApplySteering(float targetAngle)
    {
        currentSteerAngle = Mathf.Lerp(currentSteerAngle, targetAngle, Time.deltaTime * steeringSmoothing);
        // Debug.Log($"currentSteerAngle: {currentSteerAngle}");
        FLC.steerAngle = currentSteerAngle;
        FRC.steerAngle = currentSteerAngle;
    }

    private void ApplyMotorTorque(float targetTorque)
    {
        currentMotorTorque = Mathf.Lerp(currentMotorTorque, targetTorque, Time.deltaTime * accelerationSmoothing);
        if (finishingPointDetected && targetTorque == 0f)
            currentMotorTorque = 0;
        // Debug.Log($"currentMotorTorque: {currentMotorTorque}");
        FLC.motorTorque = currentMotorTorque;
        FRC.motorTorque = currentMotorTorque;
        RLC.motorTorque = currentMotorTorque;
        RRC.motorTorque = currentMotorTorque;
    }

    private void UpdateWheelTransforms()
    {
        UpdateWheelTransform(FLC, FLT);
        UpdateWheelTransform(FRC, FRT);
        UpdateWheelTransform(RLC, RLT);
        UpdateWheelTransform(RRC, RRT);
    }

    private void UpdateWheelTransform(WheelCollider collider, Transform wheelTransform)
    {
        Vector3 position;
        Quaternion rotation;
        collider.GetWorldPose(out position, out rotation);
        wheelTransform.position = position;
        wheelTransform.rotation = rotation;
    }

}
