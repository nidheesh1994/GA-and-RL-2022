using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;

public class RobotAgent : Agent
{

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
    private float episodeTime = 0f;


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

        transform.localPosition = new Vector3(-95.3f, 34.5f, -241f); // second position for GA
        transform.rotation = Quaternion.Euler(0f, 3.969f, -0.001f);

        // Set sensor orientations as defined
        SetSensorOrientations();
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

    public override void CollectObservations(VectorSensor sensor)
    {
        // Ensure you're passing 10 floats here in the same order used in training
        // Example: 7 raycast distances + ORS + ORSZ + speed
        var sensorReadings = GetSensorData();
        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);

        Debug.Log("Collect");

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

    public override void OnActionReceived(ActionBuffers actions)
    {
        float motorInput = actions.ContinuousActions[0];
        float steerInput = actions.ContinuousActions[1];

        float motorTorque = motorInput * 500f;  // same as training
        float steerAngle = steerInput * 40f;

        // Apply to your robot
        ManualApplyControl(motorTorque, steerAngle);
    }

    public void ManualApplyControl(float torque, float steering)
    {
        ApplySteering(steering);
        ApplyMotorTorque(torque);
        // Debug.Log($"Torque: {torque}, steering: {currentSteerAngle}, car: {individualIndex}");
        UpdateWheelTransforms();
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
