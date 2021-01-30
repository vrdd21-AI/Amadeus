package com.example.amadeus;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.RelativeLayout;

public class MainActivity extends AppCompatActivity {
    private RelativeLayout relativeLayout;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        relativeLayout = findViewById(R.id.amadeus_layout);
        relativeLayout.setVisibility(View.GONE);
    }

    public void Clicked(View view){
        Log.d("test", "Clicked!");
        //Intent intent = new Intent(this, TestActivity.class);
        //startActivity(intent);
    }


    public void RineClicked(View view){
        Intent intent = new Intent(this, ChatRoomActivity.class);
        startActivityForResult(intent, 1);
    }

    public void AmadeusClicked(View view){
        Intent intent = new Intent(this, AmadeusActivity.class);
        startActivity(intent);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode == 10){
            relativeLayout.setVisibility(View.VISIBLE);
        }
    }
}
