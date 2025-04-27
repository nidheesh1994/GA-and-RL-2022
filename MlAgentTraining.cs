using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

public class MlAgentTraining : Agent
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
    public float motorTorque = 2000f;
    public float maxSpeed = 50f;
    public float turnSpeed = 30f;

    [Header("Sensor Parameters")]
    public float sensorRange = 10f;
    public float obstacleDetectionDistance = 3f;
    public string roadMaterial = "MT_Road_01";

    [Header("Performance Tuning")]
    public float steeringSmoothing = 5f;
    public float accelerationSmoothing = 2f;

    private float currentSteerAngle = 0f;
    private float currentMotorTorque = 0f;

    private float episodeTime = 0f; // Track the elapsed time since the episode started
    private const float maxEpisodeTime = 25000f; // Max episode time in seconds (120 seconds)
    private Vector3 lastPosition;

    public override void OnEpisodeBegin()
    {
        // Reset the episode timer at the beginning of each episode
        episodeTime = 0f;

        // Stop the robot's velocity and angular velocity to prevent movement at the start
        Rigidbody rb = GetComponent<Rigidbody>();
        rb.velocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        // Stop all wheels from rotating by setting motorTorque to zero
        FLC.motorTorque = 0f;
        FRC.motorTorque = 0f;
        RLC.motorTorque = 0f;
        RRC.motorTorque = 0f;

        // Reset wheel steer angles to prevent unintended turning
        FLC.steerAngle = 0f;
        FRC.steerAngle = 0f;

        // transform.localPosition = new Vector3(194.7755f, 0.6679955f, -153.1348f);
        // Reset robot position and rotation as you have it in your current code

        //first position
        transform.localPosition = new Vector3(37f, 23.9f, 10.2f); // initial position for GA
        transform.rotation = Quaternion.Euler(0.088f, 181.029f, -4.497f);

        // third position
        // transform.localPosition = new Vector3(63.20256f, 14.68702f, -123.3367f);
        // transform.rotation = Quaternion.Euler(10.608f, 359.733f, -1.223f);

        // fourth position
        // transform.localPosition = new Vector3(23.57335f, 16.44876f, -23.40142f);
        // transform.rotation = Quaternion.Euler(-0.08f, 187.596f, 0.571f);

        // Set sensor orientations as defined
        SetSensorOrientations();
        lastPosition = transform.localPosition;

        // Debug.Log("Episode has started");
    }

    public override void Heuristic(in ActionBuffers actionOut)
    {
        ActionSegment<float> continuousActions = actionOut.ContinuousActions;
        continuousActions[0] = Input.GetAxisRaw("Vertical") * .5f;
        continuousActions[1] = Input.GetAxisRaw("Horizontal") * 15;
    }

    private void FixedUpdate()
    {
        RequestDecision();
    }


    public override void OnActionReceived(ActionBuffers actions)
    {
        // Update the elapsed time
        episodeTime += Time.deltaTime;
        AddReward(0.005f);

        // Debug.Log($"Episode time : {episodeTime}");

        // Check if the episode has exceeded the maximum time (120 seconds)
        if (episodeTime > maxEpisodeTime)
        {
            // End the episode after 120 seconds
            AddReward(-0.5f); // Optional: add some reward/penalty for time limit reached
            // Debug.Log("Max time passed ending");
            // Debug.Log("EndEpisode: Timeout");
            EndEpisode();
        }

        float motorTorque = actions.ContinuousActions[0] * 300f;
        float steeringAngle = actions.ContinuousActions[1] * 45f;

        // Debug.Log($"motorTorque: {motorTorque}, steeringAngle: {steeringAngle}");
        ApplySteering(steeringAngle);
        ApplyMotorTorque(motorTorque);

        UpdateWheelTransforms();

        // Reward or penalty based on track conditions
        float reward = HandleSteeringRewards(steeringAngle, motorTorque);
        AddReward(reward);

        if (IsOutOfTrack())
        {
            AddReward(-10f);
            // Debug.Log("EndEpisode: Out of track");
            EndEpisode();
        }
    }



    public float HandleSteeringRewards(float steeringAngle, float torque)
    {
        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);
        int road = GetRoad();
        float reward = 0f;

        if (speed > 0f)
        {
            reward += speed > 2f ? (speed < 6f ? 1f : -0.3f) : -1f;
        }
        else if(speed < -0.5f)
        {
            AddReward(-10f);
            // Debug.Log("EndEpisode: Out of track");
            EndEpisode();
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
        float deltaDistance = Vector3.Distance(transform.localPosition, lastPosition);
        reward += deltaDistance * 100f; // 🔁 2f is the weight — adjust as needed
        // Debug.Log($"Last position: {lastPosition}, current position: {transform.position}, deltaDistance: {deltaDistance}");

        // Update last position
        lastPosition = transform.localPosition;

        return reward;
    }

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

    private float HandleEdgeDetection()
    {
        var sensorReadings = GetSensorData();

        if (sensorReadings["Left1"].Item2.StartsWith("ED") || sensorReadings["Right1"].Item2.StartsWith("ED") || sensorReadings["Front"].Item2.StartsWith("ED"))
        {
            // Debug.Log($"left1 : {sensorReadings["Left1"].Item1}, right1: {sensorReadings["Right1"].Item1}, Front: {sensorReadings["Front"].Item1}");
            return -3f;

        }

        return 5f;
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

    private void SetSensorOrientations()
    {
        FRS.localRotation = Quaternion.Euler(8, 0, 0);
        L1S.localRotation = Quaternion.Euler(0, -15, 0);
        L2S.localRotation = Quaternion.Euler(0, -35, 0);
        L3S.localRotation = Quaternion.Euler(0, -90, 0);
        R1S.localRotation = Quaternion.Euler(0, 15, 0);
        R2S.localRotation = Quaternion.Euler(0, 35, 0);
        R3S.localRotation = Quaternion.Euler(0, 90, 0);
        Down.localRotation = Quaternion.Euler(90, 0, 0);
    }


    private void HandleCheckpointRewards(Dictionary<string, (float, string)> sensorReadings)
    {
        for (int i = 1; i <= 22; i++)
        {
            string checkpointName = "CP" + i;
            if (CheckForCheckpointPassed(checkpointName))
            {
                AddReward(0.2f * i);
                break;
            }
        }
    }

    private void HandleFinalCheckpoint()
    {
        if (HasPassedFinalCheckpoint() && IsStopped())
        {
            AddReward(1f);
            // Debug.Log("EndEpisode: Successfully completed");
            EndEpisode();
        }
    }



    private bool CheckForCheckpointPassed(string checkpointName)
    {
        // Check if the robot has passed a checkpoint
        Collider[] hitColliders = Physics.OverlapSphere(transform.position, 1f); // Adjust radius as needed
        foreach (var collider in hitColliders)
        {
            if (collider.gameObject.name.StartsWith(checkpointName))
            {
                return true; // The robot has passed this checkpoint
            }
        }
        return false;
    }

    private bool HasPassedFinalCheckpoint()
    {
        // Check if the robot has passed CP22
        return CheckForCheckpointPassed("CP22");
    }

    private bool IsStopped()
    {
        // Check if the robot has stopped moving (velocity is low)
        return GetComponent<Rigidbody>().velocity.magnitude < 0.1f; // Adjust as necessary for "stopped" state
    }

    public override void CollectObservations(VectorSensor sensor)
    {

        try
        {
            // Your observation logic here
            var sensorReadings = GetSensorData();
            float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);
            sensor.AddObservation(sensorReadings["Front"].Item1);
            sensor.AddObservation(sensorReadings["Left1"].Item1);
            sensor.AddObservation(sensorReadings["Left2"].Item1);
            sensor.AddObservation(sensorReadings["Left3"].Item1);
            sensor.AddObservation(sensorReadings["Right1"].Item1);
            sensor.AddObservation(sensorReadings["Right2"].Item1);
            sensor.AddObservation(sensorReadings["Right3"].Item1);
            sensor.AddObservation(sensorReadings["ORS"].Item1);
            sensor.AddObservation(sensorReadings["ORSZ"].Item1);
            sensor.AddObservation(speed);
            // Debug.Log("Collection");
        }
        catch (Exception ex)
        {
            // Debug.LogError($"CollectObservations Exception: {ex.Message}");
        }
    }

    private int GetHitItemId(string hitName)
    {
        if (hitName.StartsWith("MT_Road"))
            return 1;
        else if (hitName.StartsWith("MT_Turn"))
            return 2;
        else if (hitName.StartsWith("CP"))
            return 3;
        else if (hitName.StartsWith("Plane"))
            return 4;
        else if (hitName.StartsWith("ED"))
            return 5;
        else if (hitName.StartsWith("Cube"))
            return 6;
        else
            return 0;
    }


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

    private (float, string) CheckOrientationSensorZ()
    {
        // Get the robot's forward-facing angle relative to the world
        float zaw = transform.eulerAngles.z;
        // Debug.Log($"xaw pitch: {xaw}");
        // Normalize the pitch
        float normalizedPitch = (zaw > 180) ? zaw - 360 : zaw;

        // Return the xaw value along with a descriptor
        return (normalizedPitch, "OrientationZ");
    }

    private (float, string) CheckOrientationSensor()
    {
        // Get the robot's forward-facing angle relative to the world
        float xaw = transform.eulerAngles.x;
        // Debug.Log($"xaw pitch: {xaw}");
        // Normalize the pitch
        float normalizedPitch = (xaw > 180) ? xaw - 360 : xaw;

        // Return the xaw value along with a descriptor
        return (normalizedPitch, "OrientationX");
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
            Debug.DrawRay(sensor.position, sensor.forward * hit.distance, Color.red);
            return (hit.distance, hit.collider.gameObject.name);
        }
        return (sensorRange, "None");
    }

    private void ApplySteering(float targetAngle)
    {
        currentSteerAngle = Mathf.Lerp(currentSteerAngle, targetAngle, Time.deltaTime * steeringSmoothing);
        FLC.steerAngle = currentSteerAngle;
        FRC.steerAngle = currentSteerAngle;
    }

    private void ApplyMotorTorque(float targetTorque)
    {
        currentMotorTorque = Mathf.Lerp(currentMotorTorque, targetTorque, Time.deltaTime * accelerationSmoothing);
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
