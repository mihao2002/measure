// EdgeMeasureApp.swift
// MVP Prototype: Detect and measure an edge using ARKit + Vision + SwiftUI

import SwiftUI
import RealityKit
import ARKit
import Vision
import Combine

// MARK: - Shared ARView Singleton
class SharedARView: ARView {
    static let instance: SharedARView = {
        let arView = SharedARView(frame: .zero)
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]
        config.sceneReconstruction = .mesh
        arView.session.run(config)
        return arView
    }()

    private var framePublisher: AnyCancellable?
    private let visionQueue = DispatchQueue(label: "vision-edge-detection")
    var onEdgeLengthUpdate: ((Float) -> Void)?
    
    // Visual elements for edge detection feedback
    private var centerIndicatorView: UIView?
    private var measurementLineView: UIView?
    private var edgeHighlightView: UIView?
    private var contourOverlayView: UIView?
    
    // Store the last detected edge points for real-time updates
    private var lastDetectedEdgePoints: (SIMD3<Float>, SIMD3<Float>)?
    private var lastEdgeFound: Bool = false
    private var lastDetectedSegment: VNLineSegmentsObservation?

    required init(frame: CGRect) {
        super.init(frame: frame)
        self.setupFrameProcessing()
        self.setupVisualElements()
        self.setupSessionDelegate()
    }

    required init?(coder: NSCoder) { fatalError() }

    private func setupSessionDelegate() {
        session.delegate = self
    }

    private func setupFrameProcessing() {
        framePublisher = Timer.publish(every: 1.0, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in 
                self?.processCurrentFrame()
            }
    }

    private func processCurrentFrame() {
        guard let frame = session.currentFrame else { return }

        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])

        // Use line segment detection instead of contour detection
        let request = VNDetectLineSegmentsRequest()
        request.minimumLineLength = 0.1  // Minimum line length (normalized)
        request.minimumNumberOfPoints = 2

        visionQueue.async { [weak self] in
            guard let self = self else { return }
            
            do {
                try handler.perform([request])
                guard let observations = request.results?.first as? VNLineSegmentsObservation else { return }

                let center = CGPoint(x: 0.5, y: 0.5)
                
                let edgeFound = self.findStraightEdgeNearCenter(observations: observations, center: center)
                if edgeFound {
                    DispatchQueue.main.async {
                        self.raycastLength(at: CGPoint(x: self.bounds.midX, y: self.bounds.midY))
                    }
                } else {
                    // No edge found, clear stored points
                    self.lastDetectedEdgePoints = nil
                    self.lastEdgeFound = false
                    self.lastDetectedSegment = nil
                }
                
                // Store the detection state
                self.lastEdgeFound = edgeFound
                
                // Update visual feedback
                self.updateVisualFeedback(edgeFound: edgeFound)
            } catch {
                print("Vision error: \(error)")
            }
        }
    }

    private func findStraightEdgeNearCenter(observations: VNLineSegmentsObservation, center: CGPoint) -> Bool {
        // Check if any line segment has points near the center
        for segmentIndex in 0..<observations.lineSegments.count {
            let segment = observations.lineSegments[segmentIndex]
            if hasPointNearCenter(segment: segment, center: center) {
                // Store the detected segment
                lastDetectedSegment = observations
                return true
            }
        }
        
        // No segment found, clear stored segment
        lastDetectedSegment = nil
        return false
    }
    
    private func hasPointNearCenter(segment: VNLineSegmentsObservation.LineSegment, center: CGPoint) -> Bool {
        // Check if the line segment passes near the center
        let startPoint = segment.startPoint
        let endPoint = segment.endPoint
        
        // Check if either endpoint is near center, or if line passes through center
        let startNearCenter = abs(startPoint.x - Float(center.x)) < 0.05 && abs(startPoint.y - Float(center.y)) < 0.05
        let endNearCenter = abs(endPoint.x - Float(center.x)) < 0.05 && abs(endPoint.y - Float(center.y)) < 0.05
        
        return startNearCenter || endNearCenter
    }

    private func raycastLength(at screenPoint: CGPoint) {
        let results = raycast(from: screenPoint, allowing: .estimatedPlane, alignment: .any)
        guard let result1 = results.first else { return }

        let offset = CGPoint(x: screenPoint.x + 50, y: screenPoint.y)  // Increased from 10 to 50 pixels
        let results2 = raycast(from: offset, allowing: .estimatedPlane, alignment: .any)
        guard let result2 = results2.first else { return }

        let p1 = result1.worldTransform.translation
        let p2 = result2.worldTransform.translation
        let distance = simd_distance(p1, p2)

        onEdgeLengthUpdate?(distance)
        
        // Store the 3D world points for continuous updates
        lastDetectedEdgePoints = (p1, p2)
        
        // Update visual feedback with current 2D projection
        updateVisualFeedbackWithCurrentProjection()
    }

    // MARK: - Visual Feedback Methods
    private func setupVisualElements() {
        // Create center indicator (crosshair)
        centerIndicatorView = createCenterIndicator()
        if let centerView = centerIndicatorView {
            addSubview(centerView)
        }
        
        // Create measurement line overlay
        measurementLineView = createMeasurementLineView()
        if let lineView = measurementLineView {
            addSubview(lineView)
            lineView.isHidden = true
        }
        
        // Create contour overlay
        contourOverlayView = createContourOverlayView()
        if let contourView = contourOverlayView {
            addSubview(contourView)
            contourView.isHidden = true
        }
    }
    
    private func createCenterIndicator() -> UIView {
        let indicator = UIView(frame: CGRect(x: 0, y: 0, width: 40, height: 40))
        indicator.center = CGPoint(x: bounds.midX, y: bounds.midY)
        indicator.backgroundColor = UIColor.clear
        
        // Create crosshair lines
        let horizontalLine = UIView(frame: CGRect(x: 0, y: 19, width: 40, height: 2))
        horizontalLine.backgroundColor = UIColor.red
        horizontalLine.layer.cornerRadius = 1
        
        let verticalLine = UIView(frame: CGRect(x: 19, y: 0, width: 2, height: 40))
        verticalLine.backgroundColor = UIColor.red
        verticalLine.layer.cornerRadius = 1
        
        indicator.addSubview(horizontalLine)
        indicator.addSubview(verticalLine)
        
        return indicator
    }
    
    private func createMeasurementLineView() -> UIView {
        let lineView = UIView(frame: bounds)
        lineView.backgroundColor = UIColor.clear
        lineView.isUserInteractionEnabled = false
        
        // Create measurement line layer
        let lineLayer = CAShapeLayer()
        lineLayer.strokeColor = UIColor.yellow.cgColor
        lineLayer.lineWidth = 6.0  // Make it thicker
        lineLayer.lineDashPattern = [10, 5] // More visible dashed line
        lineLayer.fillColor = UIColor.clear.cgColor
        lineView.layer.addSublayer(lineLayer)
        
        return lineView
    }
    
    private func createContourOverlayView() -> UIView {
        let contourView = UIView(frame: bounds)
        contourView.backgroundColor = UIColor.clear
        contourView.isUserInteractionEnabled = false
        
        // Create contour drawing layer
        let contourLayer = CAShapeLayer()
        contourLayer.strokeColor = UIColor.cyan.cgColor
        contourLayer.lineWidth = 2.0
        contourLayer.fillColor = UIColor.clear.cgColor
        contourView.layer.addSublayer(contourLayer)
        
        return contourView
    }
    
    private func updateVisualFeedback(edgeFound: Bool, measurementPoints: (CGPoint, CGPoint)? = nil) {
        DispatchQueue.main.async {
            if edgeFound {
                // Show measurement line
                if let points = measurementPoints {
                    self.showMeasurementLine(from: points.0, to: points.1)
                }
                
                // Show detected line segment
                self.showDetectedLineSegment()
                
                // Highlight center indicator
                self.centerIndicatorView?.backgroundColor = UIColor.green.withAlphaComponent(0.3)
            } else {
                // Reset center indicator
                self.centerIndicatorView?.backgroundColor = UIColor.clear
                
                // Hide measurement line
                self.measurementLineView?.isHidden = true
                
                // Hide contour overlay
                self.contourOverlayView?.isHidden = true
            }
        }
    }
    
    private func showMeasurementLine(from startPoint: CGPoint, to endPoint: CGPoint) {
        guard let lineView = measurementLineView else { return }
        
        lineView.isHidden = false
        
        // Debug: Print the measurement points
        print("Measurement line: from \(startPoint) to \(endPoint)")
        
        // Create path for the measurement line
        let path = UIBezierPath()
        path.move(to: startPoint)
        path.addLine(to: endPoint)
        
        // Update the line layer
        if let lineLayer = lineView.layer.sublayers?.first as? CAShapeLayer {
            lineLayer.path = path.cgPath
        }
    }

    private func showDetectedLineSegment() {
        guard let contourView = contourOverlayView,
              let observations = lastDetectedSegment else { return }
        
        contourView.isHidden = false
        
        // Find the line segment that passes near the center
        let center = CGPoint(x: 0.5, y: 0.5)
        let path = UIBezierPath()
        
        for segment in observations.lineSegments {
            if hasPointNearCenter(segment: segment, center: center) {
                // Convert normalized line segment points to screen coordinates
                let startPoint = CGPoint(
                    x: CGFloat(segment.startPoint.x) * bounds.width,
                    y: CGFloat(segment.startPoint.y) * bounds.height
                )
                let endPoint = CGPoint(
                    x: CGFloat(segment.endPoint.x) * bounds.width,
                    y: CGFloat(segment.endPoint.y) * bounds.height
                )
                
                path.move(to: startPoint)
                path.addLine(to: endPoint)
                break
            }
        }
        
        // Update the contour layer
        if let contourLayer = contourView.layer.sublayers?.first as? CAShapeLayer {
            contourLayer.path = path.cgPath
        }
        
        print("Drawing line segment")
    }

    private func updateVisualFeedbackFromLastDetection() {
        if lastEdgeFound {
            updateVisualFeedbackWithCurrentProjection()
        } else {
            updateVisualFeedback(edgeFound: false)
        }
    }
    
    private func updateVisualFeedbackWithCurrentProjection() {
        guard let points = lastDetectedEdgePoints else {
            updateVisualFeedback(edgeFound: false)
            return
        }
        
        // Convert current 3D world coordinates to 2D screen coordinates
        guard let screenPoint1 = project(points.0),
              let screenPoint2 = project(points.1) else {
            updateVisualFeedback(edgeFound: false)
            return
        }
        
        // Update visual feedback with current 2D projection
        updateVisualFeedback(edgeFound: true, measurementPoints: (screenPoint1, screenPoint2))
    }
}

extension simd_float4x4 {
    var translation: SIMD3<Float> {
        return [columns.3.x, columns.3.y, columns.3.z]
    }
}

struct ARViewContainer: UIViewRepresentable {
    @Binding var edgeLength: Float

    func makeUIView(context: Context) -> SharedARView {
        let arView = SharedARView.instance
        arView.onEdgeLengthUpdate = { length in
            self.edgeLength = length
        }
        return arView
    }

    func updateUIView(_ uiView: SharedARView, context: Context) {}
}

// MARK: - ARSessionDelegate
extension SharedARView: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        // Update visual feedback on every frame for smooth tracking
        updateVisualFeedbackFromLastDetection()
    }
}


