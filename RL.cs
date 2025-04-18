using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Threading.Tasks;
using Unity.Barracuda;

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
    public NNModel onnxModelAsset;
    private Model runtimeModel;
    private IWorker worker;

    // private float torqueMean = 96.5890121459961f; first position
    // private float torqueStd = 130.80125427246094f;
    // private float steerMean = -0.21530772745609283f;
    // private float steerStd = 7.084796905517578f;

    private float torqueMean = 90.01801300048828f; // Cobined postion and best 3
    private float torqueStd = 124.83465576171875f;
    private float steerMean = 0.861481249332428f;
    private float steerStd = 6.110722064971924f;

    // private float torqueMean = 89.37027740478516f; // Cobined postion and best 5
    // private float torqueStd = 124.12162017822266f;
    // private float steerMean = 0.862391471862793f;
    // private float steerStd = 6.122593879699707f;


    public void Start()
    {
        // Reset the episode timer at the beginning of each episode
        episodeTime = 0f;

        runtimeModel = ModelLoader.Load(onnxModelAsset);
        // Create a worker to run the model
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.Auto, runtimeModel);

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


        transform.localPosition = new Vector3(34.56854f, 23.92629f, -243.2978f); // first position for GA working
        transform.rotation = Quaternion.Euler(0f, 177.441f, -0.001f);


        // transform.localPosition = new Vector3(-94.5086f, 39.55402f, -303.3212f); // seond position for GA
        // transform.rotation = Quaternion.Euler(-0.31f, 360.243f, 3.421f);

        // transform.localPosition = new Vector3(-93.75f, 34.68f, -242.83f); // third position for GA
        // transform.rotation = Quaternion.Euler(0.014f, 359.737f, 4.498f);


        // Set sensor orientations as defined
        SetSensorOrientations();
    }

    void FixedUpdate()
    {
        var sensorReadings = GetSensorData();
        float speed = Vector3.Dot(transform.forward, GetComponent<Rigidbody>().velocity);

        float[] inputData = new float[10];
        inputData[0] = sensorReadings["Front"].Item1;
        inputData[1] = sensorReadings["Left1"].Item1;
        inputData[2] = sensorReadings["Left2"].Item1;
        inputData[3] = sensorReadings["Left3"].Item1;
        inputData[4] = sensorReadings["Right1"].Item1;
        inputData[5] = sensorReadings["Right2"].Item1;
        inputData[6] = sensorReadings["Right3"].Item1;
        inputData[7] = sensorReadings["ORS"].Item1;
        inputData[8] = sensorReadings["ORSZ"].Item1;
        inputData[9] = speed;

        // Create the tensor with shape [batch, height, width, channels] = [1, 1, 1, 10]
        Tensor inputTensor = new Tensor(1, 1, 1, 10);

        // Populate the tensor with your input data
        for (int i = 0; i < 10; i++)
        {
            inputTensor[0, 0, 0, i] = inputData[i];
        }
        // Execute the model with the input tensor
        worker.Execute(inputTensor);

        // Retrieve the output tensor
        Tensor outputTensor = worker.PeekOutput();

        // Process the output as needed
        // For example, if the output is a single value:
        float normalizedTorque = outputTensor[0];
        float normalizedSteering = outputTensor[1];

        float actualTorque = normalizedTorque * torqueStd + torqueMean;
        float actualSteering = normalizedSteering * steerStd + steerMean;
        // float torque = outputTensor[0];
        // float steering = outputTensor[1];

        ManualApplyControl(actualTorque, actualSteering);

        Debug.Log($"torque: {actualTorque}, steering: {actualSteering}");

        // Dispose of tensors to free resources
        inputTensor.Dispose();
        outputTensor.Dispose();
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
        // currentSteerAngle = Mathf.Lerp(currentSteerAngle, targetAngle, Time.deltaTime * steeringSmoothing);
        // Debug.Log($"currentSteerAngle: {currentSteerAngle}");
        currentSteerAngle = Mathf.Min(40f, targetAngle); ;
        FLC.steerAngle = currentSteerAngle;
        FRC.steerAngle = currentSteerAngle;
    }

    private void ApplyMotorTorque(float targetTorque)
    {
        // currentMotorTorque = Mathf.Lerp(currentMotorTorque, targetTorque, Time.deltaTime * accelerationSmoothing);
        currentMotorTorque = Mathf.Min(300f, targetTorque);
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
