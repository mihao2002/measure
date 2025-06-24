//
//  ContentView.swift
//  measure
//
//  Created by Hao Mi on 6/24/25.
//

import SwiftUI

struct ContentView: View {
    @State private var edgeLength: Float = 0.0

    var body: some View {
        ZStack {
            ARViewContainer(edgeLength: $edgeLength)
                .edgesIgnoringSafeArea(.all)

            VStack {
                Spacer()
                Text(String(format: "Edge Length: %.2f m", edgeLength))
                    .padding()
                    .background(Color.black.opacity(0.6))
                    .foregroundColor(.white)
                    .cornerRadius(10)
                    .padding(.bottom, 30)
            }
        }
    }
}

#Preview {
    ContentView()
}
