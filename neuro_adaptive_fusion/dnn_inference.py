import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np
import os
from tflite_runtime.interpreter import Interpreter

class DNNInference(Node):
    def __init__(self):
        super().__init__("dnn_inference")

        # 1. Load Parameters
        self.declare_parameter("alpha", 5000.0)
        self.alpha = self.get_parameter("alpha").get_parameter_value().double_value

        # CRITICAL FIX: Hardcoded Absolute Path to the Installed Model
        # This bypasses all "smart" logic that was failing to find the file.
        model_path = "/root/ros2_ws/install/neuro_adaptive_fusion/share/neuro_adaptive_fusion/models/neuro_adapter.tflite"

        self.get_logger().info(f"Loading TFLite from ABSOLUTE path: {model_path}")

        try:
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get Quantization Parameters
            self.input_scale, self.input_zero = self.input_details[0]['quantization']
            self.output_scale, self.output_zero = self.output_details[0]['quantization']
            
            self.get_logger().info(f"✅ INT8 Model Loaded! Scale: {self.input_scale}")
        except Exception as e:
            self.get_logger().error(f"❌ Model Load FATAL: {e}")
            # If model fails, we MUST kill the node or it will look like it's running but do nothing
            self.destroy_node()
            return

        self.sub = self.create_subscription(Float32MultiArray, "/neuro/features", self.callback, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/neuro/covariance_correction", 10)

    def callback(self, msg):
        features = np.array(msg.data, dtype=np.float32)

        # --- PATH A: Neural Network (INT8 Quantized) ---
        # 1. Quantize: Float -> Int8
        if self.input_scale > 0:
            input_int8 = (features / self.input_scale) + self.input_zero
            input_int8 = np.clip(input_int8, -128, 127).astype(np.int8)
        else:
            input_int8 = features.astype(np.int8)

        # 2. Run Inference
        input_tensor = np.expand_dims(input_int8, axis=0)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output_int8 = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        # 3. Dequantize: Int8 -> Float
        if self.output_scale > 0:
            ai_output = (output_int8.astype(np.float32) - self.output_zero) * self.output_scale
        else:
            ai_output = output_int8.astype(np.float32)

        # --- PATH B: Safety Logic (Backup) ---
        # ZCR is feature indices 18-23. High ZCR = Vibration.
        zcr_avg = np.mean(features[18:24])
        logic_output = np.zeros_like(ai_output)
        
        # Transition Logic: If vibration > threshold, ramp up correction
        if zcr_avg > 0.05:
            logic_output[:] = zcr_avg * 2.0 

        # --- FUSION: Safety Gate ---
        # Use maximum of AI and Logic to ensure safety
        raw_correction = np.maximum(ai_output, logic_output)
        
        # Apply Gain
        final_correction = raw_correction * self.alpha

        # Logging for verification
        if np.max(final_correction) > 10.0:
            self.get_logger().info(
                f"Active! AI: {np.max(ai_output):.4f} | "
                f"Logic: {np.max(logic_output):.4f} | "
                f"Final: {np.max(final_correction):.1f}"
            )

        out_msg = Float32MultiArray()
        out_msg.data = final_correction.tolist()
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DNNInference()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
